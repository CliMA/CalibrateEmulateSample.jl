# Reference the in-tree version of CalibrateEmulateSample on Julias load path
prepend!(LOAD_PATH, [joinpath(@__DIR__, "..", "..")])

# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))
include(joinpath(@__DIR__, "GModel.jl")) # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using GaussianProcesses
using Plots
using Random
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Observations

#rng_seed = 4137
#rng_seed = 413798
rng_seed = 2413798
Random.seed!(rng_seed)


function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2)
    return norm_factor
end

function get_standardizing_factors(data::Array{FT, 1}) where {FT}
    # Input: data size: N_data*N_ensembles (splatted)
    # Ensemble median of the data
    norm_factor = median(data)
    return norm_factor
end


# Output figure save directory
example_directory = @__DIR__
println(example_directory)
figure_save_directory = joinpath(example_directory, "output")
data_save_directory = joinpath(example_directory, "output")
if !isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end

# Governing settings
# Characteristic time scale
τc = 5.0 # days, prescribed by the L96 problem
# Stationary or transient dynamics
dynamics = 2 # Transient is 2
# Statistics integration length
# This has to be less than 360 and 360 must be divisible by Ts_days
Ts_days = 30.0 # Integration length in days
# Stats type, which statistics to construct from the L96 system
# 4 is a linear fit over a batch of length Ts_days
# 5 is the mean over a batch of length Ts_days
stats_type = 5


###
###  Define the (true) parameters
###
# Define the parameters that we want to learn
F_true = 8.0 # Mean F
A_true = 2.5 # Transient F amplitude
ω_true = 2.0 * π / (360.0 / τc) # Frequency of the transient F

if dynamics == 2
    params_true = [F_true, A_true]
    param_names = ["F", "A"]
else
    params_true = [F_true]
    param_names = ["F"]
end
n_param = length(param_names)
params_true = reshape(params_true, (n_param, 1))

println(n_param)
println(params_true)


###
###  Define the parameter priors
###
# Lognormal prior or normal prior?
log_normal = false # THIS ISN't CURRENTLY IMPLEMENTED

function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2 / μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2 / μ^2)))
    return μ_log, σ_log
end
#logmean_F, logstd_F = logmean_and_logstd(F_true, 5)
#logmean_A, logstd_A = logmean_and_logstd(A_true, 0.2*A_true)

if dynamics == 2
    #prior_means = [F_true+0.5, A_true+0.5]
    prior_means = [F_true, A_true]
    prior_stds = [3.0, 0.5 * A_true]
    d1 = Parameterized(Normal(prior_means[1], prior_stds[1]))
    d2 = Parameterized(Normal(prior_means[2], prior_stds[2]))
    prior_distns = [d1, d2]
    c1 = no_constraint()
    c2 = no_constraint()
    constraints = [[c1], [c2]]
    prior_names = param_names
    priors = ParameterDistribution(prior_distns, constraints, prior_names)
else
    prior_distns = [Parameterized(Normal(F_true, 1))]
    constraints = [[no_constraint()]]
    prior_names = ["F"]
    priors = ParameterDistribution(prior_distns, constraints, prior_names)
end


###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0", "y1"]


###
###  L96 model settings
###

# Lorenz 96 model parameters
# Behavior changes depending on the size of F, N
N = 36
dt = 1 / 64.0
# Start of integration
t_start = 800.0
# Data collection length
#if dynamics==1
#    T = 2.
#else
#    T = 360. / τc
#end
T = 360.0 / τc
# Batch length
Ts = 5.0 / τc # Nondimensionalize by L96 timescale
# Integration length
Tfit = Ts_days / τc
# Initial perturbation
Fp = rand(Normal(0.0, 0.01), N);
kmax = 1
# Prescribe variance or use a number of forward passes to define true interval variability
var_prescribe = false
# Use CES to learn ω?
ω_fixed = true

# Settings
# Constructs an LSettings structure, see GModel.jl for the descriptions
lorenz_settings =
    GModel.LSettings(dynamics, stats_type, t_start, T, Ts, Tfit, Fp, N, dt, t_start + T, kmax, ω_fixed, ω_true);
lorenz_params = GModel.LParams(F_true, ω_true, A_true)

###
###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by:
###        y = G(θ) + η
###

# Lorenz forward
# Input: params: [N_params, N_ens]
# Output: gt: [N_data, N_ens]
# Dropdims of the output since the forward model is only being run with N_ens=1 
# corresponding to the truth construction
gt = dropdims(GModel.run_G_ensemble(params_true, lorenz_settings), dims = 2)

# Compute internal variability covariance
n_samples = 50
if var_prescribe == true
    n_samples = 100
    yt = zeros(length(gt), n_samples)
    noise_level = 0.05
    Γy = noise_level * convert(Array, Diagonal(gt))
    μ = zeros(length(gt))
    # Add noise
    for i in 1:n_samples
        yt[:, i] = gt .+ rand(MvNormal(μ, Γy))
    end
else
    println("Using truth values to compute covariance")
    yt = zeros(length(gt), n_samples)
    for i in 1:n_samples
        lorenz_settings_local = GModel.LSettings(
            dynamics,
            stats_type,
            t_start + T * (i - 1),
            T,
            Ts,
            Tfit,
            Fp,
            N,
            dt,
            t_start + T * (i - 1) + T,
            kmax,
            ω_fixed,
            ω_true,
        )
        yt[:, i] = GModel.run_G_ensemble(params_true, lorenz_settings_local)
    end
    # Variance of truth data
    #Γy = convert(Array, Diagonal(dropdims(mean((yt.-mean(yt,dims=1)).^2,dims=1),dims=1)))
    # Covariance of truth data
    Γy = cov(yt, dims = 2)

    # println(Γy)
    println(size(Γy), " ", rank(Γy))
end


# Construct observation object
truth = Observations.Observation(yt, Γy, data_names)
# Truth sample for EKP
truth_sample = truth.mean
#sample_ind=randperm!(collect(1:n_samples))[1]
#truth_sample = yt[:,sample_ind]
#println("Truth sample:")
#println(sample_ind)
###
###  Calibrate: Ensemble Kalman Inversion
###

# L96 settings for the forward model in the EKP
# Here, the forward model for the EKP settings can be set distinctly from the truth runs
lorenz_settings_G = lorenz_settings; # initialize to truth settings

# EKP parameters
log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

N_ens = 50 # number of ensemble members
N_iter = 5 # number of EKI iterations
# initial parameters: N_params x N_ens
initial_params = construct_initial_ensemble(priors, N_ens; rng_seed = rng_seed)

ekiobj = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_params, truth_sample, truth.obs_noise_cov, Inversion())

# EKI iterations
println("EKP inversion error:")
err = zeros(N_iter)
err_params = zeros(N_iter)
for i in 1:N_iter
    if log_normal == false
        params_i = get_u_final(ekiobj)
    else
        params_i = exp_transform(get_u_final(ekiobj))
    end
    g_ens = GModel.run_G_ensemble(params_i, lorenz_settings_G)
    EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens)
    err[i] = get_error(ekiobj)[end]
    err_params[i] = mean((params_true - mean(params_i, dims = 2)) .^ 2)
    println("Iteration: " * string(i) * ", Error (data): " * string(err[i]))
    println("Iteration: " * string(i) * ", Error (params): " * string(err_params[i]))
end

# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nEKI results:")
if log_normal == false
    println(mean(get_u_final(ekiobj), dims = 2))
else
    println(mean(exp_transform(get_u_final(ekiobj)), dims = 2))
end

###
###  Emulate: Gaussian Process Regression
###

# Emulate-sample settings
standardize = true
retained_svd_frac = 0.95

gppackage = Emulators.GPJL()
pred_type = Emulators.YType()
gauss_proc = GaussianProcess(
    gppackage;
    kernel = nothing, # use default squared exponential kernel
    prediction_type = pred_type,
    noise_learn = false,
)

# Standardize the output data
# Use median over all data since all data are the same type
yt_norm = vcat(yt...)
norm_factor = get_standardizing_factors(yt_norm)
println(size(norm_factor))
#norm_factor = vcat(norm_factor...)
norm_factor = fill(norm_factor, size(truth_sample))
println("Standardization factors")
println(norm_factor)

# Get training points from the EKP iteration number in the second input term  
N_iter = 5;
input_output_pairs = Utilities.get_training_points(ekiobj, N_iter)
normalized = true
emulator = Emulator(
    gauss_proc,
    input_output_pairs;
    obs_noise_cov = Γy,
    normalize_inputs = normalized,
    standardize_outputs = standardize,
    standardize_outputs_factors = norm_factor,
    retained_svd_frac = retained_svd_frac,
)
optimize_hyperparameters!(emulator)

# Check how well the Gaussian Process regression predicts on the
# true parameters
#if retained_svd_frac==1.0
if log_normal == false
    y_mean, y_var = Emulators.predict(emulator, reshape(params_true, :, 1), transform_to_real = true)
else
    y_mean, y_var = Emulators.predict(emulator, reshape(log.(params_true), :, 1), transform_to_real = true)
end

println("GP prediction on true parameters: ")
println(vec(y_mean))
println(" GP variance")
println(diag(y_var[1], 0))
println("true data: ")
println(truth.mean) # same, regardless of norm_factor
println("GP MSE: ")
println(mean((truth.mean - vec(y_mean)) .^ 2))
#end
###
###  Sample: Markov Chain Monte Carlo
###

# initial values
u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
println("initial parameters: ", u0)

# First let's run a short chain to determine a good step size
yt_sample = truth_sample
mcmc = MCMCWrapper(RWMHSampling(), yt_sample, priors, emulator; init_params = u0)
new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)
chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

post_mean = mean(posterior)
post_cov = cov(posterior)
println("post_mean")
println(post_mean)
println("post_cov")
println(post_cov)
println("D util")
println(det(inv(post_cov)))
println(" ")

# Plot the posteriors together with the priors and the true parameter values
if log_normal == false
    true_values = params_true
else
    true_values = log.(params_true)
end
n_params = length(true_values)
y_folder = "qoi_" * string(lorenz_settings.stats_type) * "_"
save_directory = figure_save_directory * y_folder
for idx in 1:n_params
    if idx == 1
        param = "F"
        xbounds = [F_true - 1.0, F_true + 1.0]
        xs = collect(xbounds[1]:((xbounds[2] - xbounds[1]) / 100):xbounds[2])
    elseif idx == 2
        param = "A"
        xbounds = [A_true - 1.0, A_true + 1.0]
        xs = collect(xbounds[1]:((xbounds[2] - xbounds[1]) / 100):xbounds[2])
    elseif idx == 3
        param = "ω"
        xs = collect((ω_true - 0.2):0.01:(ω_true + 0.2))
        xbounds = [xs[1], xs[end]]
    else
        throw("not implemented")
    end

    if log_normal == true
        xs = log.(xs)
        xbounds = log.(xbounds)
    end

    label = "true " * param
    posterior_distributions = get_distribution(posterior)

    histogram(
        posterior_distributions[param_names[idx]][1, :],
        bins = 100,
        normed = true,
        fill = :slategray,
        lab = "posterior",
    )
    prior_plot = get_distribution(mcmc.prior)
    # This requires StatsPlots
    plot!(xs, prior_plot[param_names[idx]], w = 2.6, color = :blue, lab = "prior")
    #plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype = "vline", w = 2.6, lab = label)
    plot!(xlims = xbounds)

    title!(param)

    figpath = joinpath(figure_save_directory, "posterior_$(param)_T_$(T)_w_$(ω_true).png")
    savefig(figpath)
    linkfig(figpath)
end

# Save data
@save data_save_directory * "input_output_pairs.jld2" input_output_pairs
