# Import modules
include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots
using GaussianProcesses
using Plots
using Random

# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Priors

rng_seed = 4137
Random.seed!(rng_seed)

# Output figure save directory
figure_save_directory = "/home/mhowland/Codes/CESPlots/figures/lorenz_96_static/"

###
###  Define the (true) parameters and their priors
###

# Define the parameters that we want to learn
F_true = 5.
param_names = ["F"]
n_param = length(param_names)

params_true = [F_true]
params_true = reshape(params_true, (1,1))
y_avg = true


# Assume lognormal priors for all three parameters
# Note: For the model G (=Cloudy) to run, N0 needs to be nonnegative, and θ 
# and k need to be positive. The EK update can result in violations of 
# these constraints - therefore, we perform CES in log space, i.e., we try 
# to find the logarithms of the true parameters (and of course, the actual
# parameters can then simply be obtained by exponentiating the final results). 
function logmean_and_logstd(u,C)
    C_log = sqrt(log(1.0 + C^2/u^2))
    u_log = sqrt(u / sqrt(1.0 + C^2/u^2))
    return u_log, C_log
end

logmean_F, logstd_F = logmean_and_logstd(5, 1)

#priors = [Priors.Prior(Normal(logmean_F, logstd_F), "F")]    # prior on F
priors = [Priors.Prior(Normal(5, 1), "F")]    # prior on F

###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0", "y1"]


###
###  Model settings
###

# Lorenz 96 model parameters
# Behavior changes depending on the size of F, N
N = 38
dt = 1/64.
tend = 1000.
nstep = Int32(tend/dt);
# Start of integration
t_start = 800.
# Integration length
#T = 15.
T = 100.
# Initial perturbation
Fp = rand(Normal(0.0, 0.01), N);

# Settings
# Constructs an LSettings structure, see GModel.jl for the descriptions
lorenz_settings = GModel.LSettings(1, t_start, T, Fp, N, dt, tend);

###
###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by:
###        y = G(θ) + η
###

# Lorenz forward
# Input: params: [N_ens, N_params]
# Output: gt: [N_ens, N_data]
# Dropdims of the output since the forward model is only being run with N_ens=1 
# corresponding to the truth construction
gt = dropdims(GModel.run_G_ensemble(params_true, lorenz_settings), dims=1)

n_samples = 100
yt = zeros(n_samples, length(gt))
noise_level = 0.05
Γy = noise_level * convert(Array, Diagonal(gt))
μ = zeros(length(gt))

# Add noise
for i in 1:n_samples
    yt[i, :] = gt .+ rand(MvNormal(μ, Γy))
end
# Construct observation object
truth = Observations.Obs(yt, Γy, data_names)


###
###  Calibrate: Ensemble Kalman Inversion
###

#log_transform(a::AbstractArray) = log.(a)
#exp_transform(a::AbstractArray) = exp.(a)

N_ens = 50 # number of ensemble members
N_iter = 5 # number of EKI iterations
# initial parameters: N_ens x N_params
initial_params = EKP.construct_initial_ensemble(N_ens, priors; rng_seed=6)
ekiobj = EKP.EKObj(initial_params, param_names, truth.mean, 
                   truth.obs_noise_cov, Inversion(), Δt=1.0)

# EKI iterations
println("EKP inversion error:")
err = zeros(N_iter) 
for i in 1:N_iter
    params_i = deepcopy(ekiobj.u[end])
    g_ens = GModel.run_G_ensemble(params_i, lorenz_settings)
    EKP.update_ensemble!(ekiobj, g_ens) 
    err[i] = mean((params_true - mean(ekiobj.u[end],dims=1)).^2)
    println("Iteration: "*string(i)*", Error: "*string(err[i]))
end

# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nEKI results:")
println(mean(deepcopy(ekiobj.u[end]), dims=1))

###
###  Emulate: Gaussian Process Regression
###

gppackage = GPEmulator.GPJL()
pred_type = GPEmulator.YType()

# Construct kernel:
# Sum kernel consisting of Matern 5/2 ARD kernel, a Squared Exponential Iso 
# kernel and white noise. Note that the kernels take the signal standard 
# deviations on a log scale as input.
len1 = 1.0
kern1 = SE(len1, 1.0)
len2 = zeros(3)
kern2 = Mat52Ard(len2, 0.0)
white = Noise(log(2.0))
# # construct kernel
GPkernel =  kern1 + kern2 + white
# Get training points from the EKP iteration number in the second input term  
u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, N_iter)
#u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, 3)
#u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, 1)
normalized = true

# Kernel defined above
#gpobj = GPEmulator.GPObj(u_tp, g_tp, gppackage; GPkernel=GPkernel, 
#			 obs_noise_cov=Sy, normalized=normalized,
#                         noise_learn=false, prediction_type=pred_type)

# Default kernel
gpobj = GPEmulator.GPObj(u_tp, g_tp, gppackage; 
			 obs_noise_cov=Γy, normalized=normalized,
                         noise_learn=false, prediction_type=pred_type)

# Check how well the Gaussian Process regression predicts on the
# true parameters
y_mean, y_var = GPEmulator.predict(gpobj, reshape(params_true, 1, :), 
                                   transform_to_real=true)

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
u0 = vec(mean(u_tp, dims=1))
println("initial parameters: ", u0)

# MCMC parameters    
mcmc_alg = "rwm" # random walk Metropolis

# First let's run a short chain to determine a good step size
burnin = 0
step = 0.1 # first guess
max_iter = 5000
yt_sample = truth.mean
mcmc_test = MCMC.MCMCObj(yt_sample, Γy, priors, step, u0, max_iter, 
                         mcmc_alg, burnin, svdflag=true)
new_step = MCMC.find_mcmc_step!(mcmc_test, gpobj)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)
u0 = vec(mean(u_tp, dims=1))

# reset parameters 
burnin = 1000
max_iter = 100000

mcmc = MCMC.MCMCObj(yt_sample, Γy, priors, new_step, u0, max_iter, 
                    mcmc_alg, burnin, svdflag=true)
MCMC.sample_posterior!(mcmc, gpobj, max_iter)

posterior = MCMC.get_posterior(mcmc)      

post_mean = mean(posterior, dims=1)
post_cov = cov(posterior, dims=1)
println("post_mean")
println(post_mean)
println("post_cov")
println(post_cov)
println("D util")
println(det(inv(post_cov)))
println(" ")

# Plot the posteriors together with the priors and the true parameter values
true_values = [F_true]
n_params = length(true_values)
if lorenz_settings.stats_type==1
	y_folder = "average/"
else
	y_folder = "instantaneous/"
end
save_directory = figure_save_directory*y_folder
for idx in 1:n_params
    if idx == 1
        param = "F"
	xs = collect(1:0.1:10)
    elseif idx == 2
        param = "eps"
	xs = collect(0:0.01:0.2)
    elseif idx == 3
        param = "omega"
	xs = collect(0.4:0.01:0.8)
    else
        throw("not implemented")
    end

    label = "true " * param
    histogram(posterior[:, idx], bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig(save_directory*"posterior_"*param*".png")
end
