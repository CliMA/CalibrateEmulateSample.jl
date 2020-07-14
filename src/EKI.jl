module EKI

using ..Priors
using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EKIObj
export construct_initial_ensemble
export compute_error
export update_ensemble!
export find_eki_step

"""
    EKIObj{FT<:AbstractFloat, IT<:Int}

Structure that is used in Ensemble Kalman Inversion (EKI)

#Fields
$(DocStringExtensions.FIELDS)
"""
struct EKIObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
     u::Vector{Array{FT, 2}}
     "vector of parameter names"
     unames::Vector{String}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "vector of arrays of size N_ensemble x N_data containing the data G(u) (in each EKI iteration a new array of data is added)"
     g::Vector{Array{FT, 2}}
     "vector of errors"
     err::Vector{FT}
     "vector of timesteps used in each EKI iteration"
     Δt::Vector{FT}
end

# outer constructors
function EKIObj(parameters::Array{FT, 2},
                parameter_names::Vector{String},
                t_mean,
                t_cov::Array{FT, 2};
                Δt=FT(1)) where {FT<:AbstractFloat}

    # ensemble size
    N_ens = size(parameters)[1]
    IT = typeof(N_ens)
    # parameters
    u = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(u, parameters) # insert parameters at end of array (in this case just 1st entry)
    # observations
    g = Vector{FT}[]
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    EKIObj{FT,IT}(u, parameter_names, t_mean, t_cov, N_ens, g, err, Δt)
end


"""
construct_initial_ensemble(N_ens::IT, priors::Array{Prior, 1}; rng_seed=42) where {IT<:Int}

Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::IT, priors::Array{Prior, 1}; rng_seed=42) where {IT<:Int}
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i].dist
        if !(typeof(priors[i].dist) == Deterministic{Float64})
            params[:, i] = rand(prior_i, N_ens)
        else
            params[:, i] = prior_i.value * ones(N_ens)
        end
    end

    return params
end

function compute_error(eki)
    meang = dropdims(mean(eki.g[end], dims=1), dims=1)
    diff = eki.g_t - meang
    X = eki.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eki.err, newerr)
end


function update_ensemble!(eki::EKIObj{FT}, g; cov_threshold::FT=0.01, Δt_new=nothing) where {FT}
    # u: N_ens x N_params
    u = eki.u[end]
    cov_init = cov(eki.u[end], dims=1)

    u_bar = fill(FT(0), size(u)[2])
    # g: N_ens x N_data
    g_bar = fill(FT(0), size(g)[2])

    cov_ug = fill(FT(0), size(u)[2], size(g)[2])
    cov_gg = fill(FT(0), size(g)[2], size(g)[2])

    if !isnothing(Δt_new)
        push!(eki.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(eki.Δt)
        push!(eki.Δt, FT(1))
    else
        push!(eki.Δt, eki.Δt[end])
    end

    # update means/covs with new param/observation pairs u, g
    for j = 1:eki.N_ens

        u_ens = u[j, :]
        g_ens = g[j, :]

        # add to mean
        u_bar += u_ens
        g_bar += g_ens

        #add to cov
        cov_ug += u_ens * g_ens' # cov_ug is N_params x N_data
        cov_gg += g_ens * g_ens'
    end

    u_bar = u_bar / eki.N_ens
    g_bar = g_bar / eki.N_ens
    cov_ug = cov_ug / eki.N_ens - u_bar * g_bar'
    cov_gg = cov_gg / eki.N_ens - g_bar * g_bar'

    # update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]), eki.cov/eki.Δt[end]), eki.N_ens) # N_data * N_ens
    y = (eki.g_t .+ noise)' # add g_t (N_data) to each column of noise (N_data x N_ens), then transp. into N_ens x N_data
    tmp = (cov_gg + eki.cov) \ (y - g)' # N_data x N_data \ [N_ens x N_data - N_ens x N_data]' --> tmp is N_data x N_ens
    u += (cov_ug * tmp)' # N_ens x N_params

    # store new parameters (and observations)
    push!(eki.u, u) # N_ens x N_params
    push!(eki.g, g) # N_ens x N_data

    compute_error(eki)

    # Check convergence
    cov_new = cov(eki.u[end], dims=1)
    cov_ratio = det(cov_new)/det(cov_init)
    if cov_ratio < cov_threshold
        @warn string("New ensemble covariance determinant is less than ",cov_threshold," times its former value.
            Consider reducing the EKI time step.")
    end
end


"""
   find_eki_step(eki::EKIObj{FT}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}
Find largest step for the EKI solver that leads to a reduction of the determinant of the sample  
covariance matrix no greater than cov_threshold.
"""
function find_eki_step(eki::EKIObj{FT}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}
    accept_step = false
    if !isempty(eki.Δt)
        Δt = deepcopy(eki.Δt[end])
    else
        Δt = FT(1)
    end
    # u: N_ens x N_params
    cov_init = cov(eki.u[end], dims=1)
    while accept_step == false
        eki_copy = deepcopy(eki)
        update_ensemble!(eki_copy, g, Δt_new=Δt)
        cov_new = cov(eki_copy.u[end], dims=1)
        if det(cov_new) > cov_threshold*det(cov_init)
            accept_step = true
        else
            Δt = Δt/2
        end
    end

    return Δt

end

end # module EKI
