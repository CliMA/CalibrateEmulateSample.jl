# # Linear model with parameter transformations

# This example applies the CalibrateEmulateSample procedure to a simple linear model
# ``y_m = g(θ) = Gθ``, where ``θ ∈ R^2`` and ``y_m ∈ R^3``. All parameters in ``θ``
# are required to remain positive-definite, so the CalibrateEmulateSample procedure 
# is applied in log-transformed space.

# Import modules
using Distributions
using StatsBase
using LinearAlgebra
using GaussianProcesses
using StatsPlots
using Plots

# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.EKI
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Priors

"""
    logmean_and_logstd(μ, σ)

Returns the lognormal parameters μ and σ from the mean μ and std σ of the 
lognormal distribution.
"""
function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2/μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2/μ^2)))
    return μ_log, σ_log
end

"""
    mean_and_std_from_ln(μ, σ)

Returns the mean and variance of the lognormal distribution
from the lognormal parameters μ and σ.
"""
function mean_and_std_from_ln(μ_log, σ_log)
    μ = exp(μ_log + σ_log^2/2)
    σ = sqrt( (exp(σ_log^2) - 1)* exp(2*μ_log + σ_log^2) )
    return μ, σ
end

exp_transform(a::AbstractArray) = exp.(a)


###
###  1. Define the parameters and their priors
###

# Define the parameters that we want to learn
param_names = ["a", "b"]
n_param = length(param_names)

# Assume lognormal priors
# Note: For some dynamic models, all parameters need to be nonnegative. 
# The EKI update can result in violations of 
# these constraints - therefore, we perform CES in log space.

# Prior: Log-normal in original space defined by mean and std
logmeans = zeros(n_param)
logstds = zeros(n_param)
logmeans[1], logstds[1] = logmean_and_logstd(0.5, 0.3)
logmeans[2], logstds[2] = logmean_and_logstd(0.5, 0.3)
println("Lognmean and logstd of the prior: ", logmeans[1], " ", logstds[1])
priors = [Priors.Prior(Normal(logmeans[1], logstds[1]), "a"),    # prior on a
          Priors.Prior(Normal(logmeans[2], logstds[2]), "b" )]    # prior on b


###
###  2. Define the observations
###

# This is the true value of the observables. 
# Typically we only have access to noisy samples of these.
n_observables = 3
data_names = ["Obs_1", "Obs_2", "Obs_3"]
yt = [1.0, 2.0, 3.0]

# This is how many noisy samples of the true data we have
n_samples = 100
samples = zeros(n_samples, length(yt))
# Noise level of the samples
noise_level = 0.01
Γy = noise_level^2 * convert(Array, Diagonal(ones(3)))
μ_noise = zeros(length(yt))

# Create artificial samples by adding gaussian noise
for i in 1:n_samples
    samples[i, :] = yt + rand(MvNormal(μ_noise, Γy))
end

# We construct the observations object with the samples and the cov.
truth = Observations.Obs(samples, Γy, data_names)


###
###  3. Define the dynamic model
###

# This is the definition of the model we want to calibrate and 
# emulate. This step is not needed for practical applications since
# the model would be external to the code.

# Define linear model dynamic update g(θ) = Gθ
G = [1.0 1.0; 4.0 -2.0; 9.0/2.0 0.0]
g(x) = G*x


###
###  Calibrate: Ensemble Kalman Inversion
###

N_ens = 40 # number of ensemble members
# With EKI, using more iterations may result in a poorly trained GP for this problem!
N_iter = 2 # number of EKI iterations.

# initial parameters: N_ens x N_params
initial_params = EKI.construct_initial_ensemble(N_ens, priors)
ekiobj = EKI.EKIObj(initial_params, param_names, truth.mean, truth.obs_noise_cov)
g_ens = zeros(N_ens, n_observables)

# Estimate EKI step to ensure covariance is not negligible
params_i = deepcopy(exp_transform(ekiobj.u[end]))
for j in 1:N_ens
      g_ens[j, :] = g(params_i[j, :])
    end
Δt = EKI.find_eki_step(ekiobj, g_ens, cov_threshold=0.1)
println("EKI timestep set to: ", Δt)

# EKI iterations
for i in 1:N_iter
    # Note that the parameters are exp-transformed back to the 
    # original space for use as input in the dynamic update.
    params_i = deepcopy(exp_transform(ekiobj.u[end]))
    for j in 1:N_ens
      g_ens[j, :] = g(params_i[j, :])
    end
    # The EKI object contains the parameters in log-transformed space.
    EKI.update_ensemble!(ekiobj, g_ens; Δt_new=Δt)
end

# EKI results: Has the ensemble collapsed toward the truth?
# The true value of the parameters for this problem is
params_true = [2.0/3.0 1.0/3.0]
println("True parameters (original space): ")
println(params_true)

println("\nEKI ensemble mean at last stage (original space):")
println(mean(deepcopy(exp_transform(ekiobj.u[end])), dims=1))

println("\nEnsemble covariance from last stage of EKI, in transformed space.")
println(cov(deepcopy((ekiobj.u[end])), dims=1))


###
###  Emulate: Gaussian Process Regression
###

# Use the julia GP package
gppackage = GPEmulator.GPJL()
pred_type = GPEmulator.YType()

# Get training points: Use all stages (EKI)
u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, N_iter)

normalized = true
gpobj = GPEmulator.GPObj(u_tp, g_tp, gppackage;
                         normalized=normalized, prediction_type=pred_type)

# Check how well the Gaussian Process regression predicts on the
# true parameters
y_mean, y_var = GPEmulator.predict(gpobj, reshape(log.(params_true), 1, :))
println("\nTrue output: ")
println(truth.mean)
println("\nGP output prediction on true parameters: ")
println(vec(y_mean))

# Check output for other parameters to check GP landscape
params_other = [1.0/3.0 2.0/3.0]
y_other, y_var_other = GPEmulator.predict(gpobj, reshape(params_other, 1, :))
println("\nModel output for suboptimal parameters: ")
println(g(params_other')')
println("\nGP output prediction on same parameters: ")
println(vec(y_other))


###
###  Sample: Markov Chain Monte Carlo
###

# initial values: use ensemble mean of last EKI iteration
u0 = vec(mean(deepcopy(ekiobj.u[end]), dims=1))
println("initial parameters for MCMC (log-transformed): ", u0)

mcmc_alg = "rwm" # random walk Metropolis

# First let's run a short chain to determine a good step size
burnin = 0
step = 1.0e-1 # first guess
max_iter = 5000
yt_sample = truth.mean
mcmc_test = MCMC.MCMCObj(yt_sample, truth.obs_noise_cov, 
                         priors, step, u0, 
                         max_iter, mcmc_alg, burnin)
new_step = MCMC.find_mcmc_step!(mcmc_test, gpobj)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)

# reset parameters
max_iter = 500000
burnin = 25000

# u0 has been modified by MCMCObj in the test, so we reset it to the EKI ensemble mean.
u0 = vec(mean(ekiobj.u[end], dims=1))

mcmc = MCMC.MCMCObj(yt_sample, truth.obs_noise_cov, priors, 
                    new_step, u0, max_iter, mcmc_alg, burnin)
MCMC.sample_posterior!(mcmc, gpobj, max_iter)

posterior = MCMC.get_posterior(mcmc)      

post_mean = mean(posterior, dims=1)
println("\nPosterior mean (log-transformed)")
println(post_mean)
println("\nPosterior mean (original)")
println(exp_transform(post_mean))

# Plot the posteriors together with the priors and the 
#   true parameter values in transformed space
true_values = [log(params_true[1]) log(params_true[2])]
n_params = length(true_values)

plot(posterior[:, 1], posterior[:, 2], seriestype = :scatter)
savefig("MCMC_chain_transformed.png")

for idx in 1:n_params
    param = param_names[idx]
    label = "true " * param
    xs = collect(
        logmeans[idx]-3.0*logstds[idx]:logstds[idx]/20.0:logmeans[idx]+3.0*logstds[idx])
    histogram(posterior[:, idx], bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end

# Plot the posteriors together with the priors and the 
#   true parameter values in the original space
true_values = [(params_true[1]) (params_true[2])]

plot(exp_transform(posterior[:, 1]), exp_transform(posterior[:, 2]), seriestype = :scatter)
savefig("MCMC_chain_original.png")

for idx in 1:n_params
    param = param_names[idx]
    label = "true " * param
    xs = collect(
        logmeans[idx]-3.0*logstds[idx]:logstds[idx]/20.0:logmeans[idx]+3.0*logstds[idx])
    histogram(exp_transform(posterior[:, idx]), bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    priors_ln = [Distributions.LogNormal(logmeans[1], logstds[1]),    # prior on a
          Distributions.LogNormal(logmeans[1], logstds[2])]    # prior on b

    plot!(exp_transform(xs), priors_ln[idx], w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_orig_space_"*param*".png")
end

