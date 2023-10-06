using DocStringExtensions

using Random
using Distributions
using LinearAlgebra
using FFTW

"""
GSettings{FT<:AbstractFloat, KT, D}
Structure to hold all information to run the forward model G
# Fields
$(DocStringExtensions.FIELDS)
"""
struct GSettings{FT <: AbstractFloat, KT, D}
    "model output"
    out::Array{FT, 1}
    "time period over which to run the model, e.g., `(0, 1)`"
    tspan::Tuple{FT, FT}
    "x domain"
    x::Array{FT, 1}
    "y domain"
    y::Array{FT, 1}
end

struct LSettings
    # Stationary or transient dynamics
    dynamics::Int32
    # G model statistics type
    stats_type::Int32
    # Integration time start
    t_start::Float64
    # Data collection length
    T::Float64
    # Integration length
    Ts::Float64
    # Duration of polynomial fit, number of samples of Ts to fit over
    Tfit::Float64
    # Initial perturbation
    Fp::Array{Float64}
    # Number of longitude steps
    N::Int32
    # Timestep
    dt::Float64
    # Simulation end time
    tend::Float64
    # For stats_type=2, number of frequencies to consider
    kmax::Int32
    # Should ω be learned using CES?
    ω_fixed::Bool
    # Truth ω
    ω_true::Float64
end

struct LParams
    # Mean forcing
    F::Float64
    # Forcing frequency for transient terms
    ω::Float64
    # Forcing amplitude for transient terms
    A::Float64
end

"""
    run_G_ensemble(params::Array{FT, 2}, settings::LSettings, rng_seed = 42) where {FT <: AbstractFloat}
Run the forward model G for an array of parameters by iteratively
calling run_G for each of the N_ensemble parameter values.
Return g_ens, an array of size N_data x N_ensemble, where
g_ens[:,j] = G(params[:,j])
 - `params` - array of size N_parameters x N_ensemble containing the
              parameters for which G will be run
 - `settings` - a GSettings struct
"""
function run_G_ensemble(params::Array{FT, 2}, settings::LSettings, rng_seed = 42) where {FT <: AbstractFloat}
    # Initilize ensemble
    N_ens = size(params, 2) # params is N_params x N_ens
    if settings.stats_type == 1
        nd = 2
    elseif settings.stats_type == 2
        nd = 1 + (2 * settings.kmax)
    elseif settings.stats_type == 3
        nd = 3
    elseif settings.stats_type == 4
        nd = Int64(2 * (settings.T / settings.Ts / settings.Tfit))
    elseif settings.stats_type == 5
        nd = Int64(settings.T / settings.Ts / settings.Tfit)
    end

    # Lorenz parameters
    #Fp = rand(Normal(0.0, 0.01), N); # Initial perturbation
    F = params[1, :] # Forcing
    if settings.dynamics == 2
        A = params[2, :] # Transience amplitude
    else
        A = F .* 0.0 # Set to zero, it is not used
    end
    if settings.ω_fixed == false
        ω = params[3, :] # Transience frequency
    end

    Random.seed!(rng_seed)

    #settings
    lorenz_params = if settings.ω_fixed == false
        [GModel.LParams(F[i], ω[i], A[i]) for i in 1:N_ens]
    elseif settings.ω_fixed == true
        [GModel.LParams(F[i], settings.ω_true, A[i]) for i in 1:N_ens]
    end

    return run_ensembles(settings, lorenz_params, nd, N_ens)
end

# Forward pass of the Lorenz 96 model
# Inputs: settings: structure with stats_type, t_start, T
# F: scalar forcing Float64(1), Fp: initial perturbation Float64(N)
# N: number of longitude steps Int64(1), dt: time step Float64(1)
# tend: End of simulation Float64(1), nstep: 
function lorenz_forward(settings::LSettings, params::LParams)
    # run the Lorenz simulation
    xn, t = lorenz_solve(settings, params)
    # Get statistics
    gt = stats(settings, xn, t)
    return gt
end

# Solve the Lorenz 96 system 
# Inputs: F: scalar forcing Float64(1), Fp: initial perturbation Float64(N)
# N: number of longitude steps Int64(1), dt: time step Float64(1)
# tend: End of simulation Float64(1), nstep: 
function lorenz_solve(settings::LSettings, params::LParams)
    # Initialize
    # nstep = Int32(ceil((settings.tend - settings.t_start) / settings.dt))
    nstep = Int32(ceil(settings.tend / settings.dt))
    xn = zeros(Float64, settings.N, nstep)
    t = zeros(Float64, nstep)
    # Initial perturbation
    X = fill(Float64(0.0), settings.N)
    X = X + settings.Fp
    Xbuffer = zeros(4, settings.N)
    # March forward in time
    for j in 1:nstep
        t[j] = settings.dt * j
        #use view to update a slice
        # RK4! modifies first and last arguments
        if settings.dynamics == 1
            RK4!(X, settings.dt, settings.N, params.F, Xbuffer)
        elseif settings.dynamics == 2
            F_local = params.F + params.A * sin(params.ω * t[j])
            RK4!(X, settings.dt, settings.N, F_local, Xbuffer)
        end
        xn[:, j] += X
    end
    # Output
    return xn, t

end

# Lorenz 96 system
# f = dx/dt overwriting first argument
# Inputs: x: state, N: longitude steps, F: forcing
function f!(xnew, x, N, F)

    # Loop over N positions
    for i in 3:(N - 1)
        xnew[i] = -x[i - 2] * x[i - 1] + x[i - 1] * x[i + 1] - x[i] + F
    end
    # Periodic boundary conditions
    xnew[1] = -x[N - 1] * x[N] + x[N] * x[2] - x[1] + F
    xnew[2] = -x[N] * x[1] + x[1] * x[3] - x[2] + F
    xnew[N] = -x[N - 2] * x[N - 1] + x[N - 1] * x[1] - x[N] + F
    # Output
    return nothing
end

# RK4 solve, overwriting first argument
function RK4!(xold, dt, N, F, buffer)
    #buffer is 4 x N zeros
    @assert size(buffer) == (4, N)

    # Predictor steps updates the final argument
    # must use view to update a slice in-place
    f!(view(buffer, 1, :), xold, N, F) #k1
    f!(view(buffer, 2, :), xold + buffer[1, :] * dt / 2.0, N, F) #k2
    f!(view(buffer, 3, :), xold + buffer[2, :] * dt / 2.0, N, F) #k3
    f!(view(buffer, 4, :), xold + buffer[3, :] * dt, N, F) #k4
    # Step
    #xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    xold .+= (dt / 6.0) * (buffer[1, :] + 2.0 * buffer[2, :] + 2.0 * buffer[3, :] + buffer[4, :])

    return nothing
end

function regression(X, y)
    beta_tilde = [ones(length(X), 1) X] \ y
    v = beta_tilde[1]
    beta = beta_tilde[2]
    return beta, v
end

function spectra(signal, Ts, t)
    # Init    
    Nt = length(t)
    if mod(Nt, 2) != 0
        t = t[1:(Nt - 1)]
        signal = signal[1:(Nt - 1)]
        Nt = Nt - 1
    end
    # Remove mean    
    u = signal .- mean(signal, dims = 1)
    # FFT   
    F = fft(u) |> fftshift
    freqs = fftfreq(length(t), 1.0 / Ts) |> fftshift
    #k = gen_k(Nt,Ts)  |> fftshift        
    T = (Nt - 1) * Ts
    A = Ts^2 / (2 * pi * T)
    uhat = A * abs.(F .^ 2)
    # Output    
    mid = Int32(Nt / 2) + 1
    f = freqs[mid:Nt]
    Fp = uhat[mid:Nt]
    return f, Fp
end

###############################
## Extract statistics
###############################
function stats(settings, xn, t)
    # Define averaging indices range
    indices = findall(x -> (x > settings.t_start) && (x < settings.tend), t)

    # Define statistics of interest
    if settings.stats_type == 1 # Mean
        # Average in time and over longitude
        gtm = mean(mean(xn[:, indices], dims = 2), dims = 1)
        # Variance
        gtc = mean(mean((xn[:, indices] .- gtm) .^ 2, dims = 2), dims = 1)
        # Combine statistics
        gt = vcat(gtm, gtc)
    elseif settings.stats_type == 2 #What about an integral under parts of the spectrum
        # Average in time and over longitude
        gtm = mean(mean(xn[:, indices], dims = 2), dims = 1)
        # Power spectra
        # Get size
        f, Fp = spectra(xn[1, indices]', settings.dt, t[indices])
        # Compute temporal spectrum for each longitude
        Fp = zeros(size(Fp, 1), settings.N)
        for i in 1:(settings.N)
            f, Fp[:, i] = spectra(xn[i, indices]', settings.dt, t[indices])
        end
        # Average spectra over periodic directions
        Fp = dropdims(mean(Fp, dims = 2), dims = 2)
        ys = partialsortperm(Fp, 1:(settings.kmax); rev = true)
        gt = vcat(gtm, Fp[ys]..., f[ys]...)
    elseif settings.stats_type == 3 # Structure function
        # Average in time and over longitude
        gtm = mean(mean(xn[:, indices], dims = 2), dims = 1)
        # Maximum
        mxval, mxind = findmax(xn[:, indices]; dims = 2)
        mxval_out = mean(mxval, dims = 1)
        # Power spectra
        # Get size
        f, Fp = spectra(xn[1, indices]', settings.dt, t[indices])
        # Compute temporal spectrum for each longitude
        Fp = zeros(size(Fp, 1), settings.N)
        for i in 1:(settings.N)
            f, Fp[:, i] = spectra(xn[i, indices]', settings.dt, t[indices])
        end
        # Average spectra over periodic directions
        Fp = dropdims(mean(Fp, dims = 2), dims = 2)
        ys = partialsortperm(Fp, 1; rev = true)
        # Period
        T = 1.0 / f[ys]
        mxval, r = findmin(abs.(t .- T); dims = 1)
        r = r[1]
        # Structure function
        xp = xn .- gtm
        st = zeros(settings.N)
        for i in 1:(settings.N)
            st[i] = mean(
                (
                    xp[1, indices[1]:(indices[length(indices)] - r)] .-
                    xp[1, (indices[1] + r):indices[length(indices)]]
                ) .^ 2,
            )
        end
        # Combine
        gt = vcat(gtm, mean(st)..., mxval_out...)
    elseif settings.stats_type == 4 # Variance sub-samples, linear fit over window
        T = settings.T
        # Calculate variance on subsamples
        # Variances over sliding windows
        Ts = settings.Ts
        N = Int64(T / Ts) # number of sliding windows
        nt = Int64(Ts / settings.dt) - 1 # length of sliding windows
        var_slide = zeros(N)
        t_slide = zeros(nt, N)
        for i in 1:N
            var_slide[i] = var(vcat(xn[:, indices[((i - 1) * nt + 1):(i * nt)]]...))
            t_slide[:, i] = t[((i - 1) * nt + 1):(i * nt)]
        end
        # Polynomial fit over a batch of sliding windows
        Tfit = Int64(settings.Tfit)
        NT = Int64(N / Tfit)
        a1 = zeros(NT)
        a2 = zeros(NT)
        t_start = zeros(NT)
        y = zeros(Tfit, NT)
        ty = zeros(Tfit, NT)
        ym = zeros(Tfit, NT)
        for i in 1:NT
            ind_low = (i - 1) * Tfit + 1
            ind_high = i * Tfit
            t_start[i] = t_slide[1, ind_low]
            ty[:, i] = t_slide[1, ind_low:ind_high] .- t_start[i]
            a1[i], a2[i] = regression(ty[:, i], var_slide[ind_low:ind_high])
            y[:, i] = a1[i] .* ty[:, i] .+ a2[i]
            ym[i] = mean(var_slide[ind_low:ind_high])
        end
        # Combine
        gt = vcat(a1..., a2...)
    elseif settings.stats_type == 5 # Variance sub-samples, mean over window
        T = settings.T
        # Calculate variance on subsamples
        # Variances over sliding windows
        Ts = settings.Ts
        N = Int64(T / Ts) # number of sliding windows
        nt = Int64(Ts / settings.dt) - 1 # length of sliding windows
        var_slide = zeros(N)
        t_slide = zeros(nt, N)
        for i in 1:N
            var_slide[i] = var(xn[:, indices[((i - 1) * nt + 1):(i * nt)]])
            t_slide[:, i] = t[((i - 1) * nt + 1):(i * nt)]
        end
        Tfit = Int64(settings.Tfit) # (a multiple of Ts)
        NT = Int64(N / Tfit)
        ym = zeros(NT)
        for i in 1:NT
            ind_low = (i - 1) * Tfit + 1
            ind_high = i * Tfit
            ym[i] = mean(var_slide[ind_low:ind_high])
        end
        # Combine
        gt = vcat(ym...)
    else
        ArgumentError("Setting " * string(settings.stats_type) * " not implemented.")
    end
    return gt
end
