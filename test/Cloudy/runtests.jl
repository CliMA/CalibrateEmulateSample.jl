# Import Cloudy modules
using Pkg; Pkg.add(PackageSpec(name="Cloudy", version="0.1.0"))
using Cloudy
const PDistributions = Cloudy.ParticleDistributions

# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots
using GaussianProcesses
using Plots
using StatsPlots

# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.EKI
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.GModel
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Priors


###
###  Define the (true) parameters and their priors
###

# Define the parameters that we want to learn
# We assume that the true particle mass distribution is a Gamma distribution 
# with parameters N0_true, θ_true, k_true
param_names = ["N0", "θ", "k"]
n_param = length(param_names)

N0_true = 300.0 
θ_true = 1.5597  
k_true = 0.0817                  
params_true = [N0_true, θ_true, k_true]
# Note that dist_true is a Cloudy distribution, not a Distributions.jl 
# distribution
dist_true = PDistributions.Gamma(N0_true, θ_true, k_true)

# Assume lognormal priors for all three parameters
# Note: For the model G (=Cloudy) to run, N0 needs to be nonnegative, and θ 
# and k need to be positive. The EKI update can result in violations of 
# these constraints - therefore, we perform CES in log space, i.e., we try 
# to find the logarithms of the true parameters (and of course, the actual
# parameters can then simply be obtained by exponentiating the final results). 
function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2/μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2/μ^2)))
    return μ_log, σ_log
end

logmean_N0, logstd_N0 = logmean_and_logstd(280., 40.)
logmean_θ, logstd_θ = logmean_and_logstd(3.0, 1.5)
logmean_k, logstd_k = logmean_and_logstd(0.5, 0.5)

priors = [Priors.Prior(Normal(logmean_N0, logstd_N0), "N0"),    # prior on N0
          Priors.Prior(Normal(logmean_θ, logstd_θ), "θ"),       # prior on θ
          Priors.Prior(Normal(logmean_k, logstd_k), "k")]       # prior on k


###
###  Define the data from which we want to learn the parameters
###

data_names = ["M0", "M1", "M2"]
moments = [0.0, 1.0, 2.0]
n_moments = length(moments)


###
###  Model settings
###

# Collision-coalescence kernel to be used in Cloudy
coalescence_coeff = 1/3.14/4
kernel_func = x -> coalescence_coeff
kernel = Cloudy.KernelTensors.CoalescenceTensor(kernel_func, 0, 100.0)

# Time period over which to run Cloudy
tspan = (0., 0.5)  


###
###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by:
###        y = G(x1, x2) + η
###

g_settings_true = GModel.GSettings(kernel, dist_true, moments, tspan)
gt = GModel.run_G(params_true, g_settings_true, PDistributions.update_params, 
                  PDistributions.moment, Cloudy.Sources.get_int_coalescence)
n_samples = 100
yt = zeros(n_samples, length(gt))
noise_level = 0.05
Γy = noise_level * convert(Array, Diagonal(gt))
μ = zeros(length(gt))

# Add noise
for i in 1:n_samples
    yt[i, :] = gt .+ rand(MvNormal(μ, Γy))
end

truth = Observations.Obs(yt, Γy, data_names)


###
###  Calibrate: Ensemble Kalman Inversion
###

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

N_ens = 50 # number of ensemble members
N_iter = 5 # number of EKI iterations
# initial parameters: N_ens x N_params
initial_params = EKI.construct_initial_ensemble(N_ens, priors; rng_seed=6)
ekiobj = EKI.EKIObj(initial_params, param_names, truth.mean, truth.cov, Δt=1.0)

# Initialize a ParticleDistribution with dummy parameters. The parameters 
# will then be set in run_G_ensemble
dummy = 1.0
dist_type = PDistributions.Gamma(dummy, dummy, dummy)
g_settings = GModel.GSettings(kernel, dist_type, moments, tspan)

# EKI iterations
for i in 1:N_iter
    # Note that the parameters are exp-transformed for use as input
    # to Cloudy
    params_i = deepcopy(exp_transform(ekiobj.u[end]))
    g_ens = GModel.run_G_ensemble(params_i, g_settings,
                                  PDistributions.update_params,
                                  PDistributions.moment,
                                  Cloudy.Sources.get_int_coalescence)
    EKI.update_ensemble!(ekiobj, g_ens) 
end

# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nEKI results:")
println(mean(deepcopy(exp_transform(ekiobj.u[end])), dims=1))


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
# regularize with white noise
white = Noise(log(2.0))
# # construct kernel
GPkernel =  kern1 + kern2 + white
# Get training points    
u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, N_iter)
normalized = true
gpobj = GPEmulator.GPObj(u_tp, g_tp, Γy, gppackage; GPkernel=GPkernel, 
                         normalized=normalized, noise_learn=false, 
                         prediction_type=pred_type)

# Check how well the Gaussian Process regression predicts on the
# true parameters
y_mean, y_var = GPEmulator.predict(gpobj, reshape(log.(params_true), 1, :), 
                                   transform_to_real=true)

println("GP prediction on true parameters: ")
println(vec(y_mean))
println("true data: ")
println(truth.mean)


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
true_values = [log(N0_true) log(θ_true) log(k_true)]
n_params = length(true_values)

for idx in 1:n_params
    if idx == 1
        param = "N0"
        xs = collect(4.5:0.01:6.5)
    elseif idx == 2
        param = "Theta"
        xs = collect(-1.0:0.01:2.5)
    elseif idx == 3
        param = "k"
        xs = collect(-4.0:0.01:1.0)
    else
        throw("not implemented")
    end

    label = "true " * param
    histogram(posterior[:, idx], bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end
