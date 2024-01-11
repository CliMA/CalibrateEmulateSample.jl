using SparseArrays
using LinearAlgebra
using Printf

const σ = 5.67e-8;
const Cₛ = 1000.0 * 4000.0 * 4000 / (3600 * 24); #  ρ * c * H / seconds_per_day
const Cₐ = 1e5 / 10 * 1000 / (3600 * 24);        # Δp / g * c / seconds_per_day

struct Model{T, K, E, A, F, C, ΦF, ΦC}
	Tₛ :: T  # surface temperature
	Tₐ :: T  # atmospheric temperature
	κ  :: K  # thermal conductivity of the climate
	ε  :: E  # atmospheric emissivity
	α  :: A  # surface albedo
	Q  :: F  # forcing
	Cₛ :: C  # surface heat capacity
	Cₐ :: C  # atmospheric heat capacity
	ϕᶠ :: ΦF # the latitudinal grid at interface points (in radians)
	ϕᶜ :: ΦC # the latitudinal grid at center points (in radians)
end

function daily_insolation(lat; day = 81, S₀ = 1365.2)

	march_first = 81.0
	ϕ = deg2rad(lat)
	δ = deg2rad(23.45) * sind(360 * (day - march_first) / 365.25)

	h₀ = abs(δ) + abs(ϕ) < π / 2 ? # there is a sunset/sunrise
		 acos(-tan(ϕ) * tan(δ)) :
		 ϕ * δ > 0 ? π : 0.0 # all day or all night

	# Zenith angle corresponding to the average daily insolation
	cosθₛ = h₀ * sin(ϕ) * sin(δ) + cos(ϕ) * cos(δ) * sin(h₀)

	Q = S₀ / π * cosθₛ

	return Q
end

function annual_mean_insolation(lat; S₀ = 1365.2)
	Q_avg = 0
	for day in 1:365
		Q_avg += daily_insolation(lat; day, S₀) / 365
	end

	return Q_avg
end

Base.eltype(model::Model) = eltype(model.Tₛ)

# A simple constructor for the Model
function Model(FT = Float32;
			   ε = FT(0.8),
			   α = FT(0.2985),
			   κ = nothing,
			   N = 45,
			   Cₛ = FT(Cₛ),
			   Cₐ = FT(Cₐ),
			   initial_Tₐ = FT(250),
			   initial_Tₛ = FT(285),
			   Q = annual_mean_insolation)

	ϕᶠ = FT.(range(-π / 2, π / 2, length = N + 1))
	ϕᶜ = (ϕᶠ[2:end] .+ ϕᶠ[1:end-1]) ./ 2
	Tₛ = initial_Tₛ * ones(FT, N)
	Tₐ = initial_Tₐ * ones(FT, N)
	Q = regularize_forcing(Q, ϕᶜ)

	return Model(Tₛ, Tₐ, κ, ε, α, Q, Cₛ, Cₐ, ϕᶠ, ϕᶜ)
end

regularize_forcing(Q, ϕ) = Q
regularize_forcing(Q::Function, ϕ::Number) = eltype(ϕ)(Q(ϕ * 180 / π))
regularize_forcing(Q::Function, ϕ::AbstractArray) = eltype(ϕ).(Q.(ϕ .* 180 / π))

# A pretty show method that displays the model's parameters
function Base.show(io::IO, model::Model)
	print(io, "One-D climate model with:", '\n',
		"├── ε: ", show_parameter(emissivity(model)), '\n',
		"├── α: ", show_parameter(albedo(model)), '\n',
		"├── κ: ", show_parameter(diffusivity(model)), '\n',
		"└── Q: ", show_parameter(model.Q), " Wm⁻²")
end

# Let's define functions to retrieve the properties of the model.
# It is always useful to define functions to extract struct properties so we 
# have the possibility to extend them in the future
# emissivity and albedo
show_parameter(::Nothing)        = @sprintf("not active")
show_parameter(p::Number)        = @sprintf("%.3f", p)
show_parameter(p::AbstractArray) = @sprintf("extrema (%.3f, %.3f)", maximum(p), minimum(p))

# We define, again, the emissivities and albedo as function of the model
emissivity(model) = model.ε
emissivity(model::Model{<:Any, <:Any, <:Function}) = model.ε(model)

albedo(model) = model.α
albedo(model::Model{<:Any, <:Any, <:Any, <:Function}) = model.α(model)

diffusivity(model) = model.κ
diffusivity(model::Model{<:Any, <:Function}) = model.κ(model)

OLR(model) = σ .* ((1 .- emissivity(model)) .* model.Tₛ .^ 4 + emissivity(model) .* model.Tₐ .^ 4)
ASR(model) = (1 .- albedo(model)) .* model.Q

@inline function construct_radiative_matrix(model, Δt)
	# Temperatures at time step n
	Tₛ = model.Tₛ
	Tₐ = model.Tₐ

	ε = emissivity(model)

	Cₐ = model.Cₐ
	Cₛ = model.Cₛ

	m = length(Tₛ)

	eₐ = @. Δt * σ * Tₐ^3 * ε
	eₛ = @. Δt * σ * Tₛ^3

	# We build and insert the diagonal entries
	Da = @. Cₐ + 2 * eₐ
	Ds = @. Cₛ + eₛ

	D = vcat(Da, Ds)

	# the off-diagonal entries corresponding to the interexchange terms
	da = @. -ε * eₛ
	ds = @. -eₐ

	# spdiagm(idx => vector) constructs a sparse matrix 
	# with vector `vec` at the `idx`th diagonal 
	A = spdiagm(0 => D,
				m => da,
			   -m => ds)
	return A
end

@inline function time_step!(model, Δt)
	# Construct the LHS matrix
	A = construct_matrix(model, Δt)

	α = albedo(model)

	# Calculate the RHS
	rhsₐ = @. model.Cₐ * model.Tₐ
	rhsₛ = @. model.Cₛ * model.Tₛ + Δt * (1 - α) * model.Q

	rhs = [rhsₐ..., rhsₛ...]

	# Solve the linear system
	T = A \ rhs

	nₐ = length(model.Tₐ)
	nₛ = length(model.Tₛ)

	@inbounds @. model.Tₐ .= T[1:nₐ]
	@inbounds @. model.Tₛ .= T[nₐ+1:nₐ+nₛ]

	return nothing
end

function construct_matrix(model, Δt)

	A = construct_radiative_matrix(model, Δt)

	cosϕᶜ = cos.(model.ϕᶜ)
	Δϕ = model.ϕᶠ[2] - model.ϕᶠ[1]

	κ = diffusivity(model)

	aₛ = @. κ / Δϕ^2 / cosϕᶜ * cos(model.ϕᶠ[1:end-1])
	cₛ = @. κ / Δϕ^2 / cosϕᶜ * cos(model.ϕᶠ[2:end])

	aₐ = @. κ / Δϕ^2 / cosϕᶜ * cos(model.ϕᶠ[1:end-1])
	cₐ = @. κ / Δϕ^2 / cosϕᶜ * cos(model.ϕᶠ[2:end])

	m = length(model.Tₛ)
	for i in 1:m
		# Adding the off-diagonal entries corresponding to Tⱼ₊₁ (exclude the last row)
		if i < m
			A[i, i+1]     = -Δt * cₐ[i]
			A[i+m, i+1+m] = -Δt * cₛ[i]
		end
		# Adding the off-diagonal entries corresponding to Tⱼ₋₁ (exclude the first row)
		if i > 1
			A[i, i-1]     = -Δt * aₐ[i]
			A[i+m, i-1+m] = -Δt * aₛ[i]
		end
		# Adding the diagonal entries
		A[i, i]     += Δt * (aₐ[i] + cₐ[i])
		A[i+m, i+m] += Δt * (aₛ[i] + cₛ[i])
	end
	return A
end

# Time stepping with an implicit RHS
@inline function time_step!(model, Δt)
	# Construct the LHS matrix
	A = construct_matrix(model, Δt)

	α = albedo(model)

	# Calculate the RHS
	rhsₐ = @. model.Cₐ * model.Tₐ
	rhsₛ = @. model.Cₛ * model.Tₛ + Δt * (1 - α) * model.Q

	rhs = [rhsₐ..., rhsₛ...]

	# Solve the linear system
	T = A \ rhs

	nₐ = length(model.Tₐ)
	nₛ = length(model.Tₛ)

	@inbounds @. model.Tₐ .= T[1:nₐ]
	@inbounds @. model.Tₛ .= T[nₐ+1:nₐ+nₛ]

	return nothing
end

function evolve_model!(model; Δt = 50.0, stop_year = 1200)
	stop_iteration = ceil(Int, stop_year * 365 ÷ Δt)
	for _ in 1:stop_iteration
		time_step!(model, Δt)
	end
end

function run_ensembles(parameters, N_ens, size_obs, scale_factors)

	Ts, Os, As = scale_factors.T, scale_factors.OLR, scale_factors.ASR

	g_ens = zeros(N_ens, size_obs * 3)
	Threads.@threads for (idx, params) in enumerate(parameters)
		κₛ, α₀, α₁, ε₀, ε₁, ε₂ = params

		variable_alpha(model) = @. α₀ + α₁ * (3 * sin(model.ϕᶜ)^2 - 1)
		variable_emiss(model) = @. ε₀ + ε₁ * log2(430.0 / 280.0) + ε₂ * (model.Tₛ - 286.38)

		model = Model(; κₛ, α = variable_alpha, ε = variable_emiss)
		evolve_model!(model)

		# run the model with the current parameters, i.e., map θ to G(θ)
		g_ens[idx, :] .= [Ts .* model.Tₛ[4:end-3]...,
						  Os .* OLR(model)...,
						  As .* ASR(model)...]
	end

	return g_ens
end

