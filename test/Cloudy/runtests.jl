
# Import Cloudy modules
using Pkg; Pkg.add(PackageSpec(name="Cloudy", version="0.1.0"))
using Cloudy
const PDistributions = Cloudy.ParticleDistributions
Pkg.add("Plots")

# Import modules
using JLD2 # saving and loading Julia arrays
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots
using GaussianProcesses

# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.EKI
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.GModel
using CalibrateEmulateSample.Utilities


# Define the data from which we want to learn
data_names = ["M0", "M1", "M2"]
moments = [0.0, 1.0, 2.0]
n_moments = length(moments)
param_names = ["N0", "θ", "k"]
n_param = length(param_names)

# Define the true parameters and distribution as well as the priors of the
# parameters
N0_true = 360.
θ_true = 4.1
k_true = 2.0
u_true = [N0_true, θ_true, k_true]
priors = [Distributions.Normal(300., 50.),   # prior on N0
          Distributions.Normal(6.0, 2.0),    # prior on θ
          Distributions.Normal(3.0, 1.5)]    # prior on k

# We assume that the true particle mass distribution is a Gamma distribution
# with parameters N0_true, θ_true, k_true. Note that dist_true has to be a
# Cloudy ParticleDistribution, not a "regular" Julia distribution.
dist_true = PDistributions.Gamma(N0_true, θ_true, k_true)

# Collision-coalescence kernel to be used in Cloudy
coalescence_coeff = 1/3.14/4
kernel_func = x -> coalescence_coeff
kernel = Cloudy.KernelTensors.CoalescenceTensor(kernel_func, 0, 100.0)

# Time period over which to run Cloudy
tspan = (0., 0.5)


###
###  Generate (artifical) truth samples
###

g_settings_true = GModel.GSettings(kernel, dist_true, moments, tspan)
yt = GModel.run_G(u_true, g_settings_true,
                  PDistributions.update_params,
                  PDistributions.moment,
                  Cloudy.Sources.get_int_coalescence)
n_samples = 100
samples = zeros(n_samples, length(yt))
noise_level = 0.05
Γy = noise_level^2 * convert(Array, Diagonal(yt))
μ = zeros(length(yt))
for i in 1:n_samples
    samples[i, :] = yt + noise_level^2 * rand(MvNormal(μ, Γy))
end
truth = Observations.Obs(samples, Γy, data_names)


###
###  Calibrate: Ensemble Kalman Inversion
###

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

N_ens = 50 # number of ensemble members
N_iter = 5 # number of EKI iterations
# initial parameters: N_ens x N_param
initial_param = EKI.construct_initial_ensemble(N_ens, priors; rng_seed=5)
# Note: For the model G (=Cloudy) to run, N0 needs to be nonnegative, and θ and
# k need to be positive. The EKI update can result in violations of these
# constraints - therefore, we log-transform the initial ensemble, perform all
# EKI operations in log space and run G with the exponentiated parameters,
# which ensures positivity.
ekiobj = EKI.EKIObj(log_transform(initial_param),
                    param_names, truth.mean, truth.cov)

# Initialize a ParticleDistribution with dummy parameters. The parameters will
# then be set in run_model_ensemble
dummy = 1.0
dist_type = PDistributions.Gamma(dummy, dummy, dummy)

g_settings = GModel.GSettings(kernel, dist_type, moments, tspan)
# EKI iterations
for i in 1:N_iter
    # Note the exp-transform to ensure positivity of the
    # parameters
    g_ens = GModel.run_G_ensemble(exp_transform(ekiobj.u[end]),
                                  g_settings,
                                  PDistributions.update_params,
                                  PDistributions.moment,
                                  Cloudy.Sources.get_int_coalescence
                                  )
    EKI.update_ensemble!(ekiobj, g_ens)
end

# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(u_true)
println("\nEKI results:")
println(mean(exp_transform(ekiobj.u[end]), dims=1))


###
###  Emulate: Gaussian Process Regression
###

gppackage = "gp_jl" # use the GaussianProcesses.jl package

# Construct kernel:
# Sum kernel consisting of Matern 5/2 ARD kernel, a Squared Exponential Iso
# kernel and white noise. Note that the kernels take the signal standard
# deviations on a log scale as input.
len1 = 1.0
kern1 = SE(len1, 1.0)
len2 = zeros(n_param)
kern2 = Mat52Ard(len2, 0.0)
# regularize with white noise
white = Noise(log(2.0))
# construct kernel
GPkernel =  kern1 + kern2 + white
# Extract training points {u_i, G(u_i)}
log_u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, 5)
u_tp = exp_transform(log_u_tp)

# Standardize the parameters
param_mean = [mean(prior) for prior in priors]
param_std = [std(prior) for prior in priors]
u_tp_zscore = Utilities.orig2zscore(u_tp, param_mean, param_std)
# Standardize the data
data_mean = vec(mean(g_tp, dims=1))
data_std = vec(std(g_tp, dims=1))
g_tp_zscore = Utilities.orig2zscore(g_tp, data_mean, data_std)

# Fit a Gaussian Process regression to the training points
gpobj = GPEmulator.GPObj(u_tp_zscore, g_tp_zscore, gppackage, GPkernel)

# Check how well the Gaussian Process regression predicts on the
# (standardized) true parameters
u_true_zscore = Utilities.orig2zscore(u_true, param_mean, param_std)
y_mean, y_var = GPEmulator.predict(gpobj, reshape(u_true_zscore, 1, :))
y_mean = cat(y_mean..., dims=2)
y_var = cat(y_var..., dims=2)
println("GP prediction on true parameters: ")
println(vec(y_mean) .* data_std + data_mean)
println("true data: ")
println(truth.mean)


###
###  Sample: Markov Chain Monte Carlo
###

# initial values
u0 = vec(mean(u_tp, dims=1))
# MCMC parameters
mcmc_alg = "rwm" # random walk Metropolis
# First let's run a short chain to determine a good step size
burnin = 0
step = 0.1 # first guess
max_iter = 5000
standardized = true # we are using z scores
yt_sample = Utilities.get_obs_sample(truth)
yt_sample_zscore = Utilities.orig2zscore(yt_sample, data_mean, data_std)
# Normalize covariance of obs noise to determinant 1
truth_cov_norm = (1.0/(det(truth.cov)))^(1.0/n_param) * truth.cov

mcmc_test = MCMC.MCMCObj(yt_sample_zscore, truth_cov_norm,
                         priors, step, u0,
                         max_iter, mcmc_alg, burnin, standardized)
new_step = MCMC.find_mcmc_step!(mcmc_test, gpobj)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)
u0 = vec(mean(u_tp, dims=1))
# reset parameters
burnin = 1000
max_iter = 100000
mcmc = MCMC.MCMCObj(yt_sample_zscore, truth_cov_norm, priors,
                    new_step, u0, max_iter, mcmc_alg, burnin,
                    standardized)
MCMC.sample_posterior!(mcmc, gpobj, max_iter)

posterior = MCMC.get_posterior(mcmc)
post_mean = mean(posterior, dims=1)
post_cov = cov(posterior, dims=1)
println("post_mean")
println(post_mean)
println("post_cov")


# Plot the posteriors together with the priors and the true
# parameter values
for idx in 1:n_param
    if idx == 1
        param = "N0"
        xs = collect(100:1:500)
    elseif idx == 2
        param = "Theta"
        xs = collect(0:0.01:12.0)
    elseif idx == 3
        param = "k"
        xs = collect(0:0.001:6.0)
    else
        throw("not implemented")
    end

    label = "true " * param
    histogram(posterior[:, idx] .* param_std[idx] .+ param_mean[idx],
              bins=100, normed=true, fill=:slategray, lab="posterior")
    plot!(xs, mcmc.prior[idx], w=2.6, color=:blue, lab="prior")
    plot!([u_true[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end
