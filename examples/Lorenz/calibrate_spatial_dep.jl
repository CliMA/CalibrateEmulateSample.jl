include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics

# CES 
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

# G(θ) = H(Ψ(θ,x₀,t₀,t₁))
# y = G(θ) + η

# This will change for different Lorenz simulators
struct LorenzConfig{FT <: Real}
    "Length of a fixed integration timestep"
    dt::FT
    "Total duration of integration (T = N*dt)"
    T::FT
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
# - config: config of forward run
# - observation_config: config for observations

function lorenz_forward(
    params::EnsembleMemberConfig,
    x0::VV,
    config::LorenzConfig,
    observation_config::ObservationConfig,
) where {VV <: AbstractVector}
    # run the Lorenz simulation
    xn = lorenz_solve(params, x0, config)
    # Get statistics
    gt = stats(xn, config, observation_config)
    return gt
end

# Calculates statistics for forward model output
# Inputs: 
# - xn: timeseries of states for length of simulation through Lorenz96
# - config: config of forward run
# - observation_config: config for observations
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
    state_dim = size(x0, 1)
    xn = zeros(state_dim, nstep + 1)
    xn[:, 1] = x0

    # March forward in time
    for j in 1:nstep
        xn[:, j + 1] = RK4(params, xn[:, j], config)
    end

    return xn
end

# Lorenz 96 system
# f = dx/dt
# Inputs: 
# - params: structure with F (state-dependent-forcing vector) 
# - x: current state
function f(params::EnsembleMemberConfig, x::VV) where {VV <: AbstractVector}
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
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)

#Creating my sythetic data
#initalize model variables
nx = 40  #dimensions of parameter vector
gamma = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)  #forcing (Needs to be of type EnsembleMemberConfig)
true_parameters = EnsembleMemberConfig(gamma)

dt = 0.01  #time step


# Spin up over T_long for an initial condition
T_long = 1000.0  #total time 
spinup_config = LorenzConfig(dt, T_long)
x_initial = randn(rng_i, nx)
x_spun_up = lorenz_solve(true_parameters, x_initial, spinup_config) 
x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data


#Creating sythetic data
T = 14.0
ny = nx * 2   #number of data points
lorenz_config_settings = LorenzConfig(dt, T)

# construct how we compute Observations
T_start = 4.0  
T_end = T
observation_config = ObservationConfig(T_start, T_end)

model_out_y = lorenz_forward(true_parameters, x0, lorenz_config_settings, observation_config)

#Observation covariance
n_samples = 200
shuffled_ids = shuffle(Int(floor(size(x_spun_up,2)/2)):size(x_spun_up,2))
x_on_attractor = x_spun_up[:,shuffled_ids[1:n_samples]] # randomly select points from second half of spin up

y_ens = hcat(
    [
        lorenz_forward(
            true_parameters,
            x_on_attractor[:,j], 
            lorenz_config_settings,
            observation_config,
        ) for j in 1:n_samples
    ]...,
)

# estimate noise as the variability of these pushed-forward samples on the attractor
obs_noise_cov = cov(y_ens, dims = 2)
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
constraint = repeat([no_constraint()], nx)
name = "ml96_prior"

prior = ParameterDistribution(distribution, constraint, name)

########################################################################
############################# Running GNKI #############################
########################################################################

# EKP parameters
N_ens = 50
N_iter = 20

rng_seed = 2498

rng = MersenneTwister(rng_seed)

# initial parameters: N_params x N_ens
initial_params = construct_initial_ensemble(rng, prior, N_ens)

method = Inversion()

@info "Ensemble size: $(N_ens)"
ekpobj = EKP.EnsembleKalmanProcess(initial_params, y, obs_noise_cov, method; rng = copy(rng), verbose = true)

count = 0
n_iter = [0]
for i in 1:N_iter
    params_i = get_ϕ_final(prior, ekpobj)

    # If RMSE convergence criteria is not satisfied 
    G_ens = hcat(
        [
            lorenz_forward(
                EnsembleMemberConfig(params_i[:, j]),
                (x0 .+ ic_cov_sqrt * randn(rng, nx)),
                lorenz_config_settings,
                observation_config,
            ) for j in 1:N_ens
        ]...,
    )
    # Update 
    terminate = EKP.update_ensemble!(ekpobj, G_ens)
    if !isnothing(terminate)
        n_iter[1] = i - 1
        break
    end
end
final_ensemble = get_ϕ_final(prior, ekpobj)

# Output figure save directory
homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

# Create plots

gr(size = (1.6 * 400, 400))
hm = heatmap(x_spun_up[:, (end - 10000):end], c = :viridis)
savefig(hm, joinpath(figure_save_directory, "spun_up_heatmap.png"))
savefig(hm, joinpath(figure_save_directory, "spun_up_heatmap.pdf"))

using Plots.Measures
gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
p1 = plot(
    range(0, nx - 1, step = 1),
    [gamma mean(final_ensemble, dims = 2)],
    label = ["solution" "EKI"],
    color = [:black :lightgreen],
    linewidth = 4,
    xlabel = "Spatial index",
    ylabel = "Forcing (input)",
    left_margin = 15mm,
    bottom_margin = 15mm,
)

p2 = plot(
    1:length(y),
    y,
    ribbon = sqrt.(diag(get_obs_noise_cov(ekpobj))),
    label = "data",
    color = :black,
    linewidth = 4,
    xlabel = "Spatial index",
    ylabel = "State mean/std output)",
    left_margin = 15mm,
    bottom_margin = 15mm,
    xticks = (Int.(0:10:ny), [0, 10, 20, 30, (40, 0), 10, 20, 30, 40]),
)
plot!(p2, 1:length(y), get_g_mean_final(ekpobj), label = "mean-final-output", color = :lightgreen, linewidth = 4)

l = @layout [a b]
plt = plot(p1, p2, layout = l)

savefig(plt, figure_save_directory * "solution_spatial_dep_ens$(N_ens).png")

# save objects
save(joinpath(data_save_directory, "ekp_spatial_dep.jld2"), "ekpobj", ekpobj)
save(joinpath(data_save_directory, "priors_spatial_dep.jld2"), "prior", prior)

u_stored = EKP.get_u(ekpobj, return_array = false)
g_stored = EKP.get_g(ekpobj, return_array = false)


save(
    joinpath(data_save_directory, "calibrate_results_spatial_dep.jld2"),
    "inputs",
    u_stored,
    "outputs",
    g_stored,
    "truth_sample",
    y,
    "truth_sample_mean",
    mean(y_ens, dims = 2),
    "truth_input_constrained",
    gamma,
)
