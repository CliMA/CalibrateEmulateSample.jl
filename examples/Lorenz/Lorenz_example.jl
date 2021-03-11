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
using CalibrateEmulateSample.GaussianProcessEmulator
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcessModule
using CalibrateEmulateSample.ParameterDistributionStorage
using CalibrateEmulateSample.DataStorage
using CalibrateEmulateSample.Observations

rng_seed = 4137
Random.seed!(rng_seed)

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
τc = 5. # days, prescribed by the L96 problem
# Stationary or transient dynamics
dynamics = 2 # Transient is 2
# Statistics integration length
# This has to be less than 360 and 360 must be divisible by Ts_days
Ts_days = 90. # Integration length in days
# Stats type, which statistics to construct from the L96 system
# 4 is a linear fit over a batch of length Ts_days
# 5 is the mean over a batch of length Ts_days
stats_type = 5 


###
###  Define the (true) parameters
###
# Define the parameters that we want to learn
F_true = 8. # Mean F
A_true = 2.5 # Transient F amplitude
ω_true = 2. * π / (360. / τc) # Frequency of the transient F

if dynamics==2
    params_true = [F_true, A_true]
    param_names = ["F", "A"]
else
    params_true = [F_true]
    param_names = ["F"]
end
n_param = length(param_names)
params_true = reshape(params_true, (n_param,1))

println(n_param)
println(params_true)


###
###  Define the parameter priors
###
# Lognormal prior or normal prior?
log_normal = false # THIS ISN't CURRENTLY IMPLEMENTED

function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2/μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2/μ^2)))
    return μ_log, σ_log
end
#logmean_F, logstd_F = logmean_and_logstd(F_true, 5)
#logmean_A, logstd_A = logmean_and_logstd(A_true, 0.2*A_true)

if dynamics == 2
	prior_means = [F_true+1.0, A_true+0.5]
	prior_stds = [2.0, 0.5*A_true]
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
dt = 1/64.
# Start of integration
t_start = 800.
# Data collection length
if dynamics==1
    T = 2.
else
    T = 360. / τc
end
# Batch length
Ts = 5. / τc # Nondimensionalize by L96 timescale
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
lorenz_settings = GModel.LSettings(dynamics, stats_type, t_start, 
				   T, Ts, Tfit, Fp, N, dt, t_start+T, kmax,
				   ω_fixed, ω_true);
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
gt = dropdims(GModel.run_G_ensemble(params_true, lorenz_settings), dims=2)

# Compute internal variability covariance
if var_prescribe==true
    n_samples = 100
    yt = zeros(length(gt),n_samples)
    noise_level = 0.05
    Γy = noise_level * convert(Array, Diagonal(gt))
    μ = zeros(length(gt))
    # Add noise
    for i in 1:n_samples
        yt[:,i] = gt .+ rand(MvNormal(μ, Γy))
    end
else
    println("Using truth values to compute covariance")
    n_samples = 20 
    yt = zeros(length(gt), n_samples)
    for i in 1:n_samples
	    lorenz_settings_local = GModel.LSettings(dynamics, stats_type, t_start+T*(i-1), 
					      T, Ts, Tfit, Fp, N, dt, 
					      t_start+T*(i-1)+T, kmax, ω_fixed, ω_true);
	    yt[:, i] = GModel.run_G_ensemble(params_true, lorenz_settings_local)
    end
    # Variance of truth data
    #Γy = convert(Array, Diagonal(dropdims(mean((yt.-mean(yt,dims=1)).^2,dims=1),dims=1)))
    # Covariance of truth data
    Γy = cov(yt, dims=2)
    
    println(Γy)
end


# Construct observation object
truth = Observations.Obs(yt, Γy, data_names)
truth_sample = truth.mean
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
initial_params = construct_initial_ensemble(priors, N_ens; rng_seed=rng_seed)

ekiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_params, 
						           truth_sample, 
						           truth.obs_noise_cov,
						           Inversion())

# EKI iterations
println("EKP inversion error:")
err = zeros(N_iter) 
for i in 1:N_iter
    if log_normal==false
        params_i = get_u_final(ekiobj)
    else
        params_i = exp_transform(get_u_final(ekiobj))
    end
    g_ens = GModel.run_G_ensemble(params_i, lorenz_settings_G)
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens) 
    err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println("Iteration: "*string(i)*", Error: "*string(err[i]))
end

# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nEKI results:")
if log_normal==false
    println(mean(get_u_final(ekiobj), dims=2))
else
    println(mean(exp_transform(get_u_final(ekiobj)), dims=2))
end

###
###  Emulate: Gaussian Process Regression
###

gppackage = GaussianProcessEmulator.GPJL()
pred_type = GaussianProcessEmulator.YType()

# Get training points from the EKP iteration number in the second input term  
input_output_pairs = Utilities.get_training_points(ekiobj, N_iter)
normalized = true

# Default kernel
gpobj = GaussianProcessEmulator.GaussianProcess(input_output_pairs, gppackage; 
			 obs_noise_cov=Γy, normalized=normalized,
                         noise_learn=false, prediction_type=pred_type)

# Check how well the Gaussian Process regression predicts on the
# true parameters
if log_normal==false
    y_mean, y_var = GaussianProcessEmulator.predict(gpobj, reshape(params_true, :, 1), 
                                   transform_to_real=true)
else
	y_mean, y_var = GaussianProcessEmulator.predict(gpobj, reshape(log.(params_true), :, 1), 
                                   transform_to_real=true)
end

println("GP prediction on true parameters: ")
println(vec(y_mean))
println(" GP variance")
println(diag(y_var[1],0))
println("true data: ")
println(truth.mean)
println("GP MSE: ")
println(mean((truth.mean - vec(y_mean)).^2))

###
###  Sample: Markov Chain Monte Carlo
###

# initial values
u0 = vec(mean(get_inputs(input_output_pairs), dims=2))
println("initial parameters: ", u0)

# MCMC parameters    
mcmc_alg = "rwm" # random walk Metropolis

# First let's run a short chain to determine a good step size
burnin = 0
step = 0.1 # first guess
max_iter = 2000
yt_sample = truth_sample
mcmc_test = MarkovChainMonteCarlo.MCMC(yt_sample, Γy, priors, step, u0, max_iter, 
                         mcmc_alg, burnin, svdflag=true)
new_step = MarkovChainMonteCarlo.find_mcmc_step!(mcmc_test, gpobj, max_iter=max_iter)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)

# reset parameters 
burnin = 2000
max_iter = 100000

mcmc = MarkovChainMonteCarlo.MCMC(yt_sample, Γy, priors, new_step, u0, max_iter, 
                    mcmc_alg, burnin, svdflag=true)
MarkovChainMonteCarlo.sample_posterior!(mcmc, gpobj, max_iter)

println("Posterior size")
println(size(mcmc.posterior))

posterior = MarkovChainMonteCarlo.get_posterior(mcmc)      

post_mean = get_mean(posterior)
post_cov = get_cov(posterior)
println("post_mean")
println(post_mean)
println("post_cov")
println(post_cov)
println("D util")
println(det(inv(post_cov)))
println(" ")

# Plot the posteriors together with the priors and the true parameter values
if log_normal==false
    true_values = params_true
else
    true_values = log.(params_true)
end
n_params = length(true_values)
y_folder = "qoi_"*string(lorenz_settings.stats_type)*"_"
save_directory = figure_save_directory*y_folder
for idx in 1:n_params
    if idx == 1
        param = "F"
	xbounds = [7.95, 8.05]
	xs = collect(xbounds[1]:(xbounds[2]-xbounds[1])/100:xbounds[2])
    elseif idx == 2
        param = "A"
	xbounds = [2.45, 2.55]
	xs = collect(xbounds[1]:(xbounds[2]-xbounds[1])/100:xbounds[2])
    elseif idx == 3
        param = "ω"
	xs = collect(ω_true-0.2:0.01:ω_true+0.2)
	xbounds = [xs[1], xs[end]]
    else
        throw("not implemented")
    end

    if log_normal==true
        xs = log.(xs)
	xbounds = log.(xbounds)
    end

    label = "true " * param
    posterior_distributions = get_distribution(posterior)

    histogram(posterior_distributions[param_names[idx]][1,:], bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    prior_plot = get_distribution(mcmc.prior)
    # This requires StatsPlots
    plot!(xs, prior_plot[param_names[idx]], w=2.6, color=:blue, lab="prior")
    #plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)
    plot!(xlims=xbounds)

    title!(param)
    
    figpath = joinpath(figure_save_directory, "posterior_$(param)_T_$(T)_w_$(ω_true).png")
    savefig(figpath)
    linkfig(figpath)
end

# Save data
@save data_save_directory*"input_output_pairs.jld2" input_output_pairs
