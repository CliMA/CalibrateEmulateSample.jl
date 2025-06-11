include("../../Lorenz/GModel.jl") # Contains Lorenz 96 source code

include("./forward_maps.jl")

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics

# CES 
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

# G(θ) = H(Ψ(θ,x₀,t₀,t₁))
# y = G(θ) + η

# This will change for different Lorenz simulators
struct LorenzConfig{FT1 <: Real, FT2 <: Real}
    "Length of a fixed integration timestep"
    dt::FT1
    "Total duration of integration (T = N*dt)"
    T::FT2
end

# This will change for each ensemble member
struct EnsembleMemberConfig{VV <: AbstractVector}
    "state-dependent-forcing"
    F::VV
end

# This will change for different "Observations" of Lorenz
struct ObservationConfig{FT1 <: Real, FT2 <: Real}
    "initial time to gather statistics (T_start = N_start*dt)"
    T_start::FT1
    "end time to gather statistics (T_end = N_end*dt)"
    T_end::FT2
end
#########################################################################
############################ Model Functions ############################
#########################################################################

# Forward pass of forward model
# Inputs: 
# - params: structure with F (state-dependent-forcing vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_forward(
    params::EnsembleMemberConfig,
    x0::VorM,
    config::LorenzConfig,
    observation_config::ObservationConfig,
) where {VorM <: AbstractVecOrMat}
    # run the Lorenz simulation
    xn = lorenz_solve(params, x0, config)
    # Get statistics
    gt = stats(xn, config, observation_config)
    return gt
end

#Calculates statistics for forward model output
# Inputs: 
# - xn: timeseries of states for length of simulation through Lorenz96
function stats(xn::VorM, config::LorenzConfig, observation_config::ObservationConfig) where {VorM <: AbstractVecOrMat}
    T_start = observation_config.T_start
    T_end = observation_config.T_end
    dt = config.dt
    N_start = Int(ceil(T_start / dt))
    N_end = Int(ceil(T_end / dt))
    xn_stat = xn[:, N_start:N_end]
    N_state = size(xn_stat, 1)
    gt = zeros(2 * N_state)
    gt[1:N_state] = mean(xn_stat, dims = 2)
    gt[(N_state + 1):(2 * N_state)] = std(xn_stat, dims = 2)
    return gt
end

# Forward pass of the Lorenz 96 model
# Inputs: 
# - params: structure with F (state-dependent-forcing vector)
# - x0: initial condition vector
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function lorenz_solve(params::EnsembleMemberConfig, x0::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    # Initialize    
    nstep = Int(ceil(config.T / config.dt))
    state_dim = isa(x0, AbstractVector) ? length(x0) : size(x0, 1)
    xn = zeros(size(x0, 1), nstep + 1)
    xn[:, 1] = x0

    # March forward in time
    for j in 1:nstep
        xn[:, j + 1] = RK4(params, xn[:, j], config)
    end
    # Output
    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: 
# - params: structure with F (state-dependent-forcing vector) 
# - x: current state
function f(params::EnsembleMemberConfig, x::VorM) where {VorM <: AbstractVecOrMat}
    F = params.F
    N = length(x)
    f = zeros(N)
    # Loop over N positions
    for i in 3:(N - 1)
        f[i] = -x[i - 2] * x[i - 1] + x[i - 1] * x[i + 1] - x[i] + F[i]
    end
    # Periodic boundary conditions
    f[1] = -x[N - 1] * x[N] + x[N] * x[2] - x[1] + F[1]
    f[2] = -x[N] * x[1] + x[1] * x[3] - x[2] + F[2]
    f[N] = -x[N - 2] * x[N - 1] + x[N - 1] * x[1] - x[N] + F[N]
    # Output
    return f
end

# RK4 solve
# Inputs: 
# - params: structure with F (state-dependent-forcing vector) 
# - xold: current state
# - config: structure including dt (timestep Float64(1)) and T (total time Float64(1))
function RK4(params::EnsembleMemberConfig, xold::VorM, config::LorenzConfig) where {VorM <: AbstractVecOrMat}
    N = length(xold)
    dt = config.dt

    # Predictor steps (note no time-dependence is needed here)
    k1 = f(params, xold)
    k2 = f(params, xold + k1 * dt / 2.0)
    k3 = f(params, xold + k2 * dt / 2.0)
    k4 = f(params, xold + k3 * dt)
    # Step
    xnew = xold + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Output
    return xnew
end


########################################################################
############################ Problem setup #############################
########################################################################

struct Lorenz <: ForwardMapType
    rng::Any
    config_settings::Any
    observation_config::Any
    x0::Any
    ic_cov_sqrt::Any
    nx::Any
end

# columns of X are samples
function forward_map(X::AbstractVector, model::Lorenz)
    lorenz_forward(
        EnsembleMemberConfig(X),
        (model.x0 .+ model.ic_cov_sqrt * randn(model.rng, model.nx)),
        model.config_settings,
        model.observation_config,
    )
end

function forward_map(X::AbstractMatrix, model::Lorenz)
    hcat([forward_map(x, model) for x in eachcol(X)]...)
end

function jac_forward_map(X::AbstractVector, model::Lorenz)
    # Finite-difference Jacobian
    nx = model.nx
    h = 1e-6
    J = zeros(nx * 2, nx)
    for i in 1:nx
        x_plus_h = copy(X)
        x_plus_h[i] += h
        x_minus_h = copy(X)
        x_minus_h[i] -= h
        J[:, i] = (forward_map(x_plus_h, model) - forward_map(x_minus_h, model)) / (2 * h)
    end
    return J
end

function jac_forward_map(X::AbstractMatrix, model::Lorenz)
    return [jac_forward_map(x, model) for x in eachcol(X)]
end

function lorenz(input_dim, output_dim, rng)
    #Creating my sythetic data
    #initalize model variables
    nx = 40  #dimensions of parameter vector
    ny = nx * 2   #number of data points
    @assert input_dim == nx
    @assert output_dim == ny

    gamma = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)  #forcing (Needs to be of type EnsembleMemberConfig)
    true_parameters = EnsembleMemberConfig(gamma)

    t = 0.01  #time step
    T_long = 1000.0  #total time 
    picking_initial_condition = LorenzConfig(t, T_long)

    #beginning state
    x_initial = rand(rng, Normal(0.0, 1.0), nx)

    #Find the initial condition for my data
    x_spun_up = lorenz_solve(true_parameters, x_initial, picking_initial_condition)  #Need to make LorenzConfig object with t, T_long

    #intital condition used for the data
    x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data

    #Creating my sythetic data
    T = 14.0
    lorenz_config_settings = LorenzConfig(t, T)

    # construct how we compute Observations
    T_start = 4.0  #2*max
    T_end = T
    observation_config = ObservationConfig(T_start, T_end)

    model_out_y = lorenz_forward(true_parameters, x0, lorenz_config_settings, observation_config)

    #Observation covariance
    # [Don't need to do this bit really] - initial condition perturbations
    covT = 1000.0  #time to simulate to calculate a covariance matrix of the system
    cov_solve = lorenz_solve(true_parameters, x0, LorenzConfig(t, covT))
    ic_cov = 0.1 * cov(cov_solve, dims = 2)
    ic_cov_sqrt = sqrt(ic_cov)

    n_samples = 200
    y_ens = hcat(
        [
            lorenz_forward(
                true_parameters,
                (x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, n_samples))[:, j],
                lorenz_config_settings,
                observation_config,
            ) for j in 1:n_samples
        ]...,
    )

    # estimate noise from IC-effect + R
    obs_noise_cov = cov(y_ens, dims = 2)
    y_mean = mean(y_ens, dims = 2)
    y = y_ens[:, 1]

    pl = 2.0
    psig = 3.0
    #Prior covariance
    B = zeros(nx, nx)
    for ii in 1:nx
        for jj in 1:nx
            B[ii, jj] = psig^2 * exp(-abs(ii - jj) / pl)
        end
    end
    B_sqrt = sqrt(B)

    #Prior mean
    mu = 8.0 * ones(nx)

    #Creating prior distribution
    distribution = Parameterized(MvNormal(mu, B))
    constraint = repeat([no_constraint()], 40)
    name = "ml96_prior"

    prior = ParameterDistribution(distribution, constraint, name)

    model = Lorenz(rng, lorenz_config_settings, observation_config, x0, ic_cov_sqrt, nx)

    return prior, y, obs_noise_cov, model, gamma
end
