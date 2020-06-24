# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots
using GaussianProcesses
using Plots

# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.EKI
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities


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
    logmean_and_logstd_from_mode_std(μ, σ)

Returns the lognormal parameters μ and σ from the mode and the std σ of the 
lognormal distribution.
"""
function logmean_and_logstd_from_mode_std(mode, σ)
    σ_log = sqrt( log(mode)*(log(σ^2)-1.0)/(2.0-log(σ^2)*3.0/2.0) )
    μ_log = log(mode) * (1.0-log(σ^2)/2.0)/(2.0-log(σ^2)*3.0/2.0)
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

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)


###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn
param_names = ["c_m", "c_ϵ"]
n_param = length(param_names)

# Assume lognormal priors for all three parameters
# Note: For the EDMF model to run, all parameters need to be nonnegative. 
# The EKI update can result in violations of 
# these constraints - therefore, we perform CES in log space, i.e.,
# (the parameters can then simply be obtained by exponentiating the final results). 

# Prior option 1: Log-normal in original space defined by mean and std
logmean_c_m, logstd_c_m = logmean_and_logstd(0.5, 0.3)
logmean_c_ϵ, logstd_c_ϵ = logmean_and_logstd(0.5, 0.3)
println("Lognmean and logstd of the prior: ", logmean_c_m, " ", logstd_c_m)
priors = [Distributions.Normal(logmean_c_m, logstd_c_m),    # prior on c_m
          Distributions.Normal(logmean_c_ϵ, logstd_c_ϵ)]    # prior on c_ϵ

# # Prior option 2: Log-normal in original space defined by mode and std
# logmean_c_m, logstd_c_m = logmean_and_logstd_from_mode_std(0.5, 0.3)
# logmean_c_ϵ, logstd_c_ϵ = logmean_and_logstd_from_mode_std(0.5, 0.3)
# println("Lognmean and logstd of the prior: ", logmean_c_m, " ", logstd_c_m)
# priors = [Distributions.Normal(logmean_c_m, logstd_c_m),    # prior on c_m
#           Distributions.Normal(logmean_c_ϵ, logstd_c_ϵ)]    # prior on c_ϵ

# Prior option 3: Uniform in transformed space
# priors = [Distributions.Uniform(-5.0, 1.0),    # prior on c_m
#           Distributions.Uniform(-5.0, 1.0)]    # prior on c_ϵ


###
###  Retrieve true LES samples
###

# Define linear model dynamic update G(θ) = Aθ
A = [1.0 1.0; 4.0 -2.0; 9.0/2.0 0.0]
g(x) = A*x

# This is the true value of the observables (e.g. LES ensemble mean for EDMF)
n_observables = 3
data_names = ["Obs_1", "Obs_2", "Obs_3"]
yt = ones(n_observables)
yt[2] = 2.0
yt[3] = 3.0
# This is how many noisy samples of the true data we have
n_samples = 100
samples = zeros(n_samples, length(yt))
# Noise level of the samples
noise_level = 0.1
Γy = noise_level^2 * convert(Array, Diagonal(ones(3)))
μ_noise = zeros(length(yt))

# Create artificial samples by sampling noise
for i in 1:n_samples
    samples[i, :] = yt + rand(MvNormal(μ_noise, Γy))
end

# We construct the observations object with the samples and the cov.
truth = Observations.Obs(samples, Γy, data_names)

###
###  Calibrate: Ensemble Kalman Inversion
###

N_ens = 80 # number of ensemble members
# Using EKI, more than 1-2 iterations results in a poorly trained GP for this problem!
N_iter = 2 # number of EKI iterations. 


# initial parameters: N_ens x N_params
initial_params = EKI.construct_initial_ensemble(N_ens, priors)
ekiobj = EKI.EKIObj(initial_params, param_names, truth.mean, truth.cov)

g_ens = zeros(N_ens, n_observables)
# EKI iterations
for i in 1:N_iter
    # Note that the parameters are exp-transformed for use as input
    # to SCAMPy
    params_i = deepcopy(exp_transform(ekiobj.u[end]))
    for j in 1:N_ens
      g_ens[j, :] = g(params_i[j, :])
    end
    # run_SCAMPy(params_i, ...)  ### This is not yet implemented, 
            ### it returns the predicted output y_scm = G(θ)
    EKI.update_ensemble!(ekiobj, g_ens)
end

# EKI results: Has the ensemble collapsed toward the truth?
# The true value of the parameters for this problem is
params_true = [2.0/3.0 1.0/3.0]
println("True parameters: ")
println(params_true)

println("\nEKI ensemble mean at last stage:")
println(mean(deepcopy(exp_transform(ekiobj.u[end])), dims=1))

println("\nEnsemble covariance from last stage of EKI, in transformed space.")
println(cov(deepcopy((ekiobj.u[end])), dims=1))


###
###  Emulate: Gaussian Process Regression
###

gppackage = GPEmulator.GPJL()
pred_type = GPEmulator.YType()

# Get training points: Use ensemble at last step, or all stages?
# For EKI, better to use all stages
u_tp, g_tp = Utilities.extract_GP_tp(ekiobj, N_iter)
normalized = true
gpobj = GPEmulator.GPObj(u_tp, g_tp, gppackage;
                         normalized=normalized, prediction_type=pred_type)

# Check how well the Gaussian Process regression predicts on the
# true parameters
y_mean, y_var = GPEmulator.predict(gpobj, reshape(log.(params_true), 1, :))
# other parameters to check variance of GP
y_other, y_var_other = GPEmulator.predict(gpobj, reshape(([1.0/3.0 2.0/3.0]), 1, :))
println("true data: ")
println(truth.mean)
println("GP prediction on true parameters: ")
println(vec(y_mean))
println("GP prediction on suboptimal parameters: ")
println(vec(y_other))

###
###  Sample: Markov Chain Monte Carlo
###

# initial values: use ensemble mean of last EKI iteration
u0 = vec(mean(deepcopy(ekiobj.u[end]), dims=1))
println("initial parameters for MCMC (log-transformed): ", u0)

# MCMC parameters    
mcmc_alg = "rwm" # random walk Metropolis

# First let's run a short chain to determine a good step size
burnin = 0
step = 1.0e-1 # first guess
max_iter = 5000
yt_sample = truth.mean
mcmc_test = MCMC.MCMCObj(yt_sample, truth.cov, 
                         priors, step, u0, 
                         max_iter, mcmc_alg, burnin)
new_step = MCMC.find_mcmc_step!(mcmc_test, gpobj)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)

# reset parameters
burnin = 10000
max_iter = 500000

# u0 has been modified by MCMCObj in the test, resetting.
u0 = vec(mean(ekiobj.u[end], dims=1))

mcmc = MCMC.MCMCObj(yt_sample, truth.cov, priors, 
                    new_step, u0, max_iter, mcmc_alg, burnin)
MCMC.sample_posterior!(mcmc, gpobj, max_iter)

posterior = MCMC.get_posterior(mcmc)      

post_mean = mean(posterior, dims=1)
post_cov = cov(posterior, dims=1)
println("posterior mean and variance")
println(mean_and_std_from_ln(post_mean[1], post_cov[1,1]))
println(mean_and_std_from_ln(post_mean[2], post_cov[2,2]))
println("post_cov")
println(post_cov)
println("D util")
println(det(inv(post_cov)))
println(" ")


using StatsPlots; 
using Plots;

# Plot the posteriors together with the priors and the 
#   true parameter values in transformed space
true_values = [log(params_true[1]) log(params_true[2])]
n_params = length(true_values)

plot(posterior[:, 1], posterior[:, 2], seriestype = :scatter)
savefig("MCMC_chain_transformed.png")

for idx in 1:n_params
    if idx == 1
        param = "c_m"
        xs = collect(-5.0:0.05:1.0)
    elseif idx == 2
        param = "c_ϵ"
        xs = collect(-5.0:0.05:1.0)
    end

    label = "true " * param
    histogram(posterior[:, idx], bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    plot!(xs, mcmc.prior[idx], w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end

# Plot the posteriors together with the priors and the 
#   true parameter values in the original space
true_values = [(params_true[1]) (params_true[2])]

plot(exp.(posterior[:, 1]), exp.(posterior[:, 2]), seriestype = :scatter)
savefig("MCMC_chain_original.png")

for idx in 1:n_params
    if idx == 1
        param = "c_m"
        xs = collect(-5.0:0.05:1.0)
    elseif idx == 2
        param = "c_ϵ"
        xs = collect(-5.0:0.05:1.0)
    end

    label = "true " * param
    histogram(exp.(posterior[:, idx]), bins=100, normed=true, fill=:slategray, 
              lab="posterior")
    priors_ln = [Distributions.LogNormal(logmean_c_m, logstd_c_m),    # prior on c_m
          Distributions.LogNormal(logmean_c_ϵ, logstd_c_ϵ)]    # prior on c_ϵ

    plot!(exp.(xs), priors_ln[idx], w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_orig_space_"*param*".png")
end
