
# Import Cloudy modules
using Pkg; Pkg.add(PackageSpec(url="https://github.com/climate-machine/Cloudy.jl"))
using Cloudy
using Cloudy.KernelTensors
PDistributions = Cloudy.Distributions
Pkg.add("Plots")

# Import modules
using JLD2 # saving and loading Julia arrays
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
using StatsPlots

# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.EKI
using CalibrateEmulateSample.Truth
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC


# Define the data from which we want to learn
data_names = ["M0", "M1", "M2"]
moments = [0.0, 1.0, 2.0]
n_moments = length(moments)
param_names = ["N0", "θ", "k"]

FT = Float64

# Define the true parameters and distribution as well as the
# priors of the parameters
N0_true = 3650.
θ_true = 4.1
k_true = 2.0
priors = [Distributions.Normal(3000., 500.), # prior on N0
          Distributions.Normal(6.0, 2.0),    # prior on θ
          Distributions.Normal(3.0, 1.5)]    # prior on k

get_src = Cloudy.Sources.get_int_coalescence
# We assume that the true particle mass distribution is a
# Gamma distribution with parameters N0_true, θ_true, k_true
# Note that dist_true has to be a Cloudy distribution, not a
# "regular" Julia distribution
dist_true = PDistributions.Gamma(N0_true, θ_true, k_true)

# Collision-coalescence kernel to be used in Cloudy
coalescence_coeff = 1/3.14/4
# kernel = ConstantCoalescenceTensor(coalescence_coeff)
kernel_func = x -> coalescence_coeff
kernel = CoalescenceTensor(kernel_func, 0, 100.0)

# Time period over which to run Cloudy
tspan = (0., 0.5)

# Generate the truth (a Truth.TruthObj)
truth = Truth.run_model_truth(kernel,
                               dist_true,
                               moments,
                               tspan,
                               data_names,
                               nothing,
                               PDistributions.moment,
                               get_src)

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

N_ens = 50 # number of ensemble members
N_iter = 5 # number of EKI iterations
# initial parameters: N_ens x N_params
initial_params = EKI.construct_initial_ensemble(N_ens, priors; rng_seed=5)
# Note: For the model G (=Cloudy) to run, N0 needs to be
# nonnegative, and θ and k need to be positive.
# The EKI update can result in violations of these constraints -
# therefore, we log-transform the initial ensemble, perform all
# EKI operations in log space and run G with the exponentiated
# parameters, which ensures positivity.
ekiobj = EKI.EKIObj(log_transform(initial_params),
                    param_names, truth.y_t, truth.cov)

# Initialize a CDistribution with dummy parameters
# The parameters will then be set in run_model_ensemble
dummy = 1.0
dist_type = PDistributions.Gamma(dummy, dummy, dummy)

# EKI iterations
for i in 1:N_iter
    # Note the exp-transform to ensure positivity of the
    # parameters
    g_ens = EKI.run_model_ensemble(kernel,
                                   dist_type,
                                   exp_transform(ekiobj.u[end]),
                                   moments,
                                   tspan,
                                   PDistributions.moment,
                                   PDistributions.update_params,
                                   get_src)
    EKI.update_ensemble!(ekiobj, g_ens)
end

# EKI results: Has the ensemble collapsed toward the truth?
true_params = PDistributions.get_params(truth.distr_init)
println("True parameters: ")
println(true_params)

println("\nEKI results:")
println(mean(exp_transform(ekiobj.u[end]), dims=1))

using Plots
gr()

# p = plot(exp_transform(ekiobj.u[1][:, 2]),
#          exp_transform(ekiobj.u[1][:, 3]),
#          seriestype=:scatter)
# for i in 2:N_iter
#     plot!(exp_transform(ekiobj.u[i][:, 2]),
#           exp_transform(ekiobj.u[i][:, 3]),
#           seriestype=:scatter)
# end

# plot!([θ_true], xaxis="theta", yaxis="k", seriestype="vline",
#       linestyle=:dash, linecolor=:red)
# plot!([k_true], seriestype="hline", linestyle=:dash, linecolor=:red)


# savefig(p, "EKI_test.png")

gppackage = "gp_jl"
yt, yt_cov, yt_covinv, log_u_tp, g_tp = GPEmulator.extract(truth, ekiobj, N_iter)
u_tp = exp_transform(log_u_tp)

# Standardize both the data and the parameters
data_mean = vec(mean(g_tp, dims=1))
data_std = vec(std(g_tp, dims=1))
g_tp_zscore = GPEmulator.orig2zscore(g_tp, data_mean, data_std)

param_mean = [mean(prior) for prior in priors]
param_std = [std(prior) for prior in priors]
u_tp_zscore = GPEmulator.orig2zscore(u_tp, param_mean, param_std)

# Fit a Gaussian Process regression to the training points
gpobj = GPEmulator.emulate(u_tp_zscore, g_tp_zscore, gppackage)

# Check how well the Gaussian Process regression predicts on the
# (standardized) true parameters
u_true_zscore = GPEmulator.orig2zscore([N0_true θ_true k_true],
                                        param_mean, param_std)

y_mean, y_var = GPEmulator.predict(gpobj, reshape(u_true_zscore, 1, :))
y_mean = cat(y_mean..., dims=2)
y_var = cat(y_var..., dims=2)

println("GP prediction on true parameters: ")
println(vec(y_mean) .* data_std + data_mean)
println("true data: ")
println(truth.y_t)

# initial values
u0 = vec(mean(u_tp_zscore, dims=1))
println("initial parameters: ", u0)

# MCMC parameters
mcmc_alg = "rwm" # random walk Metropolis

# First let's run a short chain to determine a good step size
burnin = 0
step = 0.1 # first guess
yt_zscore = GPEmulator.orig2zscore(yt, data_mean, data_std)
#yt_cov_zscore = GPEmulator.orig2zscore(yt_cov, data_mean, data_std)

max_iter = 5000
standardized = true # we are using z scores
mcmc_test = MCMC.MCMCObj(yt_zscore, yt_cov./maximum(yt_cov),
                        priors, step, u0,
                        max_iter, mcmc_alg, burnin, standardized)
new_step, acc_ratio = MCMC.find_mcmc_step!(mcmc_test, gpobj)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)

# reset parameters
burnin = 1000
max_iter = 100000
mcmc = MCMC.MCMCObj(yt_zscore, yt_cov./maximum(yt_cov), priors,
                    new_step, u0, max_iter, mcmc_alg, burnin,
                    standardized)
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

# Plot the posteriors together with the priors and the true
# parameter values

true_values = [N0_true θ_true k_true]
n_params = length(true_values)

for idx in 1:n_params
    if idx == 1
        param = "N0"
        xs = collect(0:1:4000)
    elseif idx == 2
        param = "Theta"
        xs = collect(0:0.01:12.0)
    elseif idx == 3
        param = "k"
        xs = collect(0:0.001:6.0)
    else
        error("not implemented")
    end

    label = "true " * param
    histogram(posterior[:, idx] .* param_std[idx] .+ param_mean[idx],
              bins=100, normed=true, fill=:slategray, lab="posterior")
    plot!(xs, mcmc.prior[idx], w=2.6, color=:blue, lab="prior")
    plot!([true_values[idx]], seriestype="vline", w=2.6, lab=label)

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end
