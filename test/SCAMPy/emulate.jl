# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Transformations

using GaussianProcesses
using JLD
using Plots
using StatsPlots
###
###  Load EKP data
###
param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
    "tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]

filedir = "results_p9_n1.0_e55_i20_d632"
ekp = load(string(filedir,"/eki.jld"))
ekp_u = ekp["eki_u"]
ekp_g = ekp["eki_g"]
true_g = ekp["truth_mean"]
yt_var = ekp["truth_cov"]
# println(diag(yt_var))
println("Value of the determinant: ", det(yt_var)) # Too low for inverting matrix
ekp_err = ekp["eki_err"]

train_inds = findall(x->x>1.1 || x<0.9, (circshift(ekp_err, 1)./ekp_err))
train_inds = [3, 4]
u_train = ekp_u[train_inds]
g_train = ekp_g[train_inds]
# u_train = ekp_u[train_inds]
# g_train = ekp_g[train_inds]
u_train = cat(u_train..., dims=1)
g_train = cat(g_train..., dims=1)

println(size(u_train))
println(size(g_train))
println(size(true_g))
# All outputs non-identical to LES
outputs_to_learn = findall(true_g .!= g_train[1,:])
# Pick a couple with well-trained GPs
good_outputs = [14, 18, 130, 133, 137, 139, 147, 180, 321, 328, 332, 340, 
  470, 481, 483, 485, 487, 489, 491, 493]
outputs_to_learn = outputs_to_learn[good_outputs]
# outputs_to_learn = [16, 153, 154, 161, 162, 169, 170, 211, 212, 362, 542]
g_train = g_train[:,outputs_to_learn]
true_g = true_g[outputs_to_learn]
yt_var = yt_var[outputs_to_learn, outputs_to_learn]
# Exponentiate u to train in real space
# u_train = exp.(u_train)
println(g_train)
###
###  Emulate: Gaussian Process Regression
###

gppackage = GPEmulator.GPJL()
pred_type = GPEmulator.YType()

# Construct kernel:
# rbf kernel
rbf_len = log.(ones(size(u_train, 2)))
rbf_logstd = log(1.0)
rbf = SEArd(rbf_len, rbf_logstd)

# Matern ARD
matkern = Matern(3/2, zeros(9), 0.0)
# construct kernel
GPkernel =  matkern

normalized = true
gpobj = GPEmulator.GPObj(u_train, g_train, gppackage; GPkernel=GPkernel,
              normalized=normalized, prediction_type=pred_type, 
              non_gaussian=false, noise_learn=true)


μ_pred, σ2_pred = GPEmulator.predict(gpobj, u_train)
println(size(μ_pred), size(g_train))
x_ = range(1, size(g_train)[1], length=size(g_train)[1])

# Order by dissipation coefficient
order_index = 5
u_ = u_train[sortperm(u_train[:, order_index]), :]
x_ = exp.(u_[:, order_index])
for i in 1:size(g_train)[2]
  plot(x_, g_train[sortperm(u_[:, order_index]), i])
  plot!(x_, μ_pred[sortperm(u_[:, order_index]), i], linestyle=:dash)
  savefig(string("gaussian_process",i,".png") )
end
# Check how well the Gaussian Process regression predicts on the
# training and testing sets
y_train_sqbias = GPEmulator.gp_sqbias(gpobj, u_train, g_train)
println("Normalized mean training error:", mean(y_train_sqbias) )

###
###  Sample: Markov Chain Monte Carlo
###

# Define prior
n_param = length(param_names)
logmeans = zeros(n_param)
log_stds = zeros(n_param)
logmeans[1], log_stds[1] = logmean_and_logstd(0.2, 0.2)
logmeans[2], log_stds[2] = logmean_and_logstd(0.4, 0.2)
logmeans[3], log_stds[3] = logmean_and_logstd(2.0, 1.0)
logmeans[4], log_stds[4] = logmean_and_logstd(0.2, 0.2)
logmeans[5], log_stds[5] = logmean_and_logstd(0.2, 0.2)
logmeans[6], log_stds[6] = logmean_and_logstd(0.2, 0.2)
logmeans[7], log_stds[7] = logmean_and_logstd(0.2, 0.2)
logmeans[8], log_stds[8] = logmean_and_logstd(8.0, 1.0)
logmeans[9], log_stds[9] = logmean_and_logstd(0.2, 0.2)
priors = [Priors.Prior(Distributions.Normal(logmeans[1], log_stds[1]), param_names[1]),
              Priors.Prior(Distributions.Normal(logmeans[2], log_stds[2]), param_names[2]),
              Priors.Prior(Distributions.Normal(logmeans[3], log_stds[3]), param_names[3]),
              Priors.Prior(Distributions.Normal(logmeans[4], log_stds[4]), param_names[4]),
              Priors.Prior(Distributions.Normal(logmeans[5], log_stds[5]), param_names[5]),
              Priors.Prior(Distributions.Normal(logmeans[6], log_stds[6]), param_names[6]),
              Priors.Prior(Distributions.Normal(logmeans[7], log_stds[7]), param_names[7]),
              Priors.Prior(Distributions.Normal(logmeans[8], log_stds[8]), param_names[8]),
              Priors.Prior(Distributions.Normal(logmeans[9], log_stds[9]), param_names[9]) ]

# Prior in real space
# priors = [Priors.Prior(Distributions.Normal(0.2, 0.2), param_names[1]),
#               Priors.Prior(Distributions.Normal(0.4, 0.2), param_names[2]),
#               Priors.Prior(Distributions.Normal(2.0, 1.0), param_names[3]),
#               Priors.Prior(Distributions.Normal(0.2, 0.2), param_names[4]),
#               Priors.Prior(Distributions.Normal(0.2, 0.2), param_names[5]),
#               Priors.Prior(Distributions.Normal(0.2, 0.2), param_names[6]),
#               Priors.Prior(Distributions.Normal(0.2, 0.2), param_names[7]),
#               Priors.Prior(Distributions.Normal(8.0, 3.0), param_names[8]),
#               Priors.Prior(Distributions.Normal(0.2, 0.2), param_names[9]) ]

# priors = [Priors.Prior(Distributions.Uniform(logmeans[1] - 3.0*log_stds[1]
#   , logmeans[1] + 3.0*log_stds[1]), param_names[1]),
#                Priors.Prior(Distributions.Uniform(logmeans[2] - 3.0*log_stds[2]
#   , logmeans[2] + 3.0*log_stds[2]), param_names[2]),
#                Priors.Prior(Distributions.Uniform(logmeans[3] - 3.0*log_stds[3]
#   , logmeans[3] + 3.0*log_stds[3]), param_names[3]),
#                Priors.Prior(Distributions.Uniform(logmeans[4] - 3.0*log_stds[4]
#   , logmeans[4] + 3.0*log_stds[4]), param_names[4]),
#                Priors.Prior(Distributions.Uniform(logmeans[5] - 3.0*log_stds[5]
#   , logmeans[5] + 3.0*log_stds[5]), param_names[5]),
#                Priors.Prior(Distributions.Uniform(logmeans[6] - 3.0*log_stds[6]
#   , logmeans[6] + 3.0*log_stds[6]), param_names[6]),
#                Priors.Prior(Distributions.Uniform(logmeans[7] - 3.0*log_stds[7]
#   , logmeans[7] + 3.0*log_stds[7]), param_names[7]),
#                Priors.Prior(Distributions.Uniform(logmeans[8] - 3.0*log_stds[8]
#   , logmeans[8] + 3.0*log_stds[8]), param_names[8]),
#                Priors.Prior(Distributions.Uniform(logmeans[9] - 3.0*log_stds[9]
#   , logmeans[9] + 3.0*log_stds[9]), param_names[9]) ]

# initial values: use ensemble mean of last EKI iteration
u0 = vec(mean(deepcopy((ekp_u[end])), dims=1))
println("initial parameters for MCMC (log-transformed): ", u0)

# MCMC parameters    
mcmc_alg = "rwm" # random walk Metropolis

# First let's run a short chain to determine a good step size
burnin = 0
step = 1.0e-1 # first guess
max_iter = 5000
yt_sample = true_g
var_ones = convert(Array{Float64,2}, Diagonal(ones(size(diagm(yt_sample)))))
println("Determinant of the matrix passed to MCMC is: ", det(yt_var))
mcmc_test = MCMC.MCMCObj(yt_sample, yt_var,       # Cov matrix needs to be fixed
                         priors, step, u0, 
                         max_iter, mcmc_alg, burnin, svdflag=false)
new_step = MCMC.find_mcmc_step!(mcmc_test, gpobj)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)

# reset parameters
max_iter = 200000
burnin = 50000

# u0 has been modified by MCMCObj in the test, resetting.
u0 = vec(mean(deepcopy((ekp_u[end])), dims=1))

mcmc = MCMC.MCMCObj(yt_sample, yt_var, priors,    # Cov matrix needs to be fixed
                    new_step, u0, max_iter, mcmc_alg, burnin)
MCMC.sample_posterior!(mcmc, gpobj, max_iter)

posterior = MCMC.get_posterior(mcmc)      

post_mean = mean(posterior, dims=1)
post_cov = cov(posterior, dims=1)
println("posterior mean and variance for each parameter:")
println(mean_and_std_from_ln(post_mean[1], post_cov[1,1]))
println(mean_and_std_from_ln(post_mean[2], post_cov[2,2]))
println(mean_and_std_from_ln(post_mean[3], post_cov[3,3]))

# Save posterior to file
save("posterior.jld", "posterior", posterior)


# Plot the posteriors together with the priors and the 
#   true parameter values in transformed space
priors = [Distributions.Normal(logmeans[1], log_stds[1]),
                        Distributions.Normal(logmeans[2], log_stds[2]),
                        Distributions.Normal(logmeans[3], log_stds[3]),
                        Distributions.Normal(logmeans[4], log_stds[4]),
                        Distributions.Normal(logmeans[5], log_stds[5]),
                        Distributions.Normal(logmeans[6], log_stds[6]),
                        Distributions.Normal(logmeans[7], log_stds[7]),
                        Distributions.Normal(logmeans[8], log_stds[8]),
                        Distributions.Normal(logmeans[9], log_stds[9])]

priors_ln = [Distributions.LogNormal(logmeans[1], log_stds[1]),
                        Distributions.LogNormal(logmeans[2], log_stds[2]),
                        Distributions.LogNormal(logmeans[3], log_stds[3]),
                        Distributions.LogNormal(logmeans[4], log_stds[4]),
                        Distributions.LogNormal(logmeans[5], log_stds[5]),
                        Distributions.LogNormal(logmeans[6], log_stds[6]),
                        Distributions.LogNormal(logmeans[7], log_stds[7]),
                        Distributions.LogNormal(logmeans[8], log_stds[8]),
                        Distributions.LogNormal(logmeans[9], log_stds[9])]
for idx in 1:n_param
    param = param_names[idx]
    
    label = "true " * param
    histogram(posterior[:, idx], bins=100, normed=true, fill=:slategray, 
              lab="posterior", xlabel = param)
    xs = collect(logmeans[idx]-3.0*log_stds[idx]:log_stds[idx]/20.0:logmeans[idx]+3.0*log_stds[idx])
    plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end

# Plot the posteriors together with the priors and the 
#   true parameter values in the original space

cornerplot(exp.(posterior), compact = true, label = param_names, histpct = 0.5, color = :rainbow)
savefig("cornerplot_original.png")

corrplot(exp.(posterior), label = param_names)
savefig("corrplot_original.png")

marginalhist(exp.(posterior[:, 1]), exp.(posterior[:, 2]), 
    xlim = [0.0, 2.0], ylim = [0.0, 2.0], nbins=40, normed=true, 
    xguide = param_names[1], yguide=param_names[2], c= :bilbao)
savefig("marginalhist_original.png")

for idx in 1:n_param
    param = param_names[idx]

    label = "true " * param
    histogram(exp.(posterior[:, idx]), bins=100, normed=true, fill=:slategray, 
              lab="posterior", xlabel = param)
    

    xs = collect(logmeans[idx]-3.0*log_stds[idx]:log_stds[idx]/20.0:logmeans[idx]+3.0*log_stds[idx])
    plot!(exp.(xs), priors_ln[idx], w=2.6, color=:blue, lab="prior")

    title!(param)
    StatsPlots.savefig("orig_space_posterior_"*param*".png")
end

