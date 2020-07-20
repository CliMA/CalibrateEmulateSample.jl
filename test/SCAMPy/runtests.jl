# Import modules
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Distributions  # probability distributions and associated functions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
@everywhere using CalibrateEmulateSample
@everywhere using CalibrateEmulateSample.EKI
@everywhere using CalibrateEmulateSample.GPEmulator
@everywhere using CalibrateEmulateSample.MCMC
@everywhere using CalibrateEmulateSample.Observations
@everywhere using CalibrateEmulateSample.Utilities
@everywhere using CalibrateEmulateSample.Transformations
@everywhere using CalibrateEmulateSample.SCModel

using GaussianProcesses
using Plots
using StatsPlots
using JLD

###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn

@everywhere param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
	"tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]
@everywhere param_names_symb = param_names
@everywhere n_param = length(param_names)

# Assume lognormal priors for all three parameters
# Note: For the EDMF model to run, all parameters need to be nonnegative. 
# The EKI update can result in violations of 
# these constraints - therefore, we perform CES in log space, i.e.,
# (the parameters can then simply be obtained by exponentiating the final results). 

# Prior option 1: Log-normal in original space defined by mean and std
@everywhere logmeans = zeros(n_param)
@everywhere log_stds = zeros(n_param)
@everywhere logmeans[1], log_stds[1] = logmean_and_logstd(0.5, 0.3)
@everywhere logmeans[2], log_stds[2] = logmean_and_logstd(0.5, 0.3)
@everywhere logmeans[3], log_stds[3] = logmean_and_logstd(2.0, 1.0)
@everywhere logmeans[4], log_stds[4] = logmean_and_logstd(0.5, 0.3)
@everywhere logmeans[5], log_stds[5] = logmean_and_logstd(0.5, 0.3)
@everywhere logmeans[6], log_stds[6] = logmean_and_logstd(0.5, 0.3)
@everywhere logmeans[7], log_stds[7] = logmean_and_logstd(0.5, 0.3)
@everywhere logmeans[8], log_stds[8] = logmean_and_logstd(2.0, 1.0)
@everywhere logmeans[9], log_stds[9] = logmean_and_logstd(0.5, 0.3)
@everywhere priors = [Distributions.Normal(logmeans[1], log_stds[1]),
                        Distributions.Normal(logmeans[2], log_stds[2]),
                        Distributions.Normal(logmeans[3], log_stds[3]),
                        Distributions.Normal(logmeans[4], log_stds[4]),
                        Distributions.Normal(logmeans[5], log_stds[5]),
                        Distributions.Normal(logmeans[6], log_stds[6]),
                        Distributions.Normal(logmeans[7], log_stds[7]),
                        Distributions.Normal(logmeans[8], log_stds[8]),
                        Distributions.Normal(logmeans[9], log_stds[9])]

# # Prior option 2: Log-normal in original space defined by mode and std
# logmean_c_m, logstd_c_m = logmean_and_logstd_from_mode_std(0.5, 0.3)
# logmean_c_ϵ, logstd_c_ϵ = logmean_and_logstd_from_mode_std(0.5, 0.3)
# println("Logmean and logstd of the prior: ", logmean_c_m, " ", logstd_c_m)
# priors = [Distributions.Normal(logmean_c_m, logstd_c_m),    # prior on c_m
#           Distributions.Normal(logmean_c_ϵ, logstd_c_ϵ)]    # prior on c_ϵ

# Prior option 3: Uniform in transformed space
# @everywhere priors = [Distributions.Uniform(-5.0, 1.0),    # prior on c_m
#           Distributions.Uniform(-5.0, 1.0),
#           Distributions.Uniform(-5.0, 1.0)]    # prior on c_ϵ


###
###  Retrieve true LES samples
###

# This is the true value of the observables (e.g. LES ensemble mean for EDMF)
@everywhere ti = [10800.0, 28800.0, 10800.0, 18000.0]
@everywhere tf = [14400.0, 32400.0, 10800.0, 21600.0]
@everywhere y_names = Array{String, 1}[]
@everywhere push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
@everywhere push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
@everywhere push!(y_names, ["thetal_mean", "total_flux_h"]) #Nieuwstadt
@everywhere push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex

# Get observations
@everywhere yt = zeros(0)
@everywhere yt_var = zeros(0)
@everywhere sim_names = ["DYCOMS_RF01", "GABLS", "Nieuwstadt", "Bomex"]

@everywhere les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[1],".may20")
@everywhere sim_dir = string("Output.", sim_names[1],".00000")
@everywhere z_scm = get_profile(sim_dir, ["z_half"])
@everywhere yt_, yt_var_ = obs_LES(y_names[1], les_dir, ti[1], tf[1], z_scm = z_scm)
@everywhere append!(yt, yt_)
@everywhere append!(yt_var, yt_var_)

@everywhere les_dir = string("Output.", sim_names[2],".iles128wCov")
@everywhere sim_dir = string("Output.", sim_names[2],".00000")
@everywhere z_scm = get_profile(sim_dir, ["z_half"])
@everywhere yt_, yt_var_ = obs_LES(y_names[2], les_dir, ti[2], tf[2], z_scm = z_scm)
@everywhere append!(yt, yt_)
@everywhere append!(yt_var, yt_var_)

@everywhere les_dir = string("/groups/esm/ilopezgo/Output.Soares.dry11")
@everywhere sim_dir = string("Output.", sim_names[3],".00000")
@everywhere z_scm = get_profile(sim_dir, ["z_half"])
@everywhere yt_, yt_var_ = obs_LES(y_names[3], les_dir, ti[3], tf[3], z_scm = z_scm)
@everywhere append!(yt, yt_)
@everywhere append!(yt_var, yt_var_)

@everywhere les_dir = string("/groups/esm/ilopezgo/Output.Bomex.may18")
@everywhere sim_dir = string("Output.", sim_names[4],".00000")
@everywhere z_scm = get_profile(sim_dir, ["z_half"])
@everywhere yt_, yt_var_ = obs_LES(y_names[4], les_dir, ti[4], tf[4], z_scm = z_scm)
@everywhere append!(yt, yt_)
@everywhere append!(yt_var, yt_var_)

@everywhere n_observables = length(yt)

# This is how many samples of the true data we have
@everywhere n_samples = 1
@everywhere samples = zeros(n_samples, length(yt))
@everywhere samples[1,:] = yt
# Noise level of the samples, which scales the time variance of each output.
@everywhere noise_level = 10.0
@everywhere Γy = noise_level^2 * convert(Array, Diagonal(yt_var))
@everywhere μ_noise = zeros(length(yt))

# We construct the observations object with the samples and the cov.
@everywhere truth = Observations.Obs(samples, Γy, y_names[1])

###
###  Calibrate: Ensemble Kalman Inversion
###

@everywhere N_ens = 10 # number of ensemble members
@everywhere N_iter = 20 # number of EKI iterations.


# initial parameters: N_ens x N_param
@everywhere initial_params = EKI.construct_initial_ensemble(N_ens, priors)
@everywhere ekiobj = EKI.EKIObj(initial_params, param_names, truth.mean, truth.cov)
@everywhere g_ens = zeros(N_ens, n_observables)

@everywhere scm_dir = "/home/ilopezgo/SCAMPy/"
@everywhere params_i = deepcopy(exp_transform(ekiobj.u[end]))

@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, param_names,
   y_names, scm_dir, ti, tf)

# EKI iterations
@everywhere Δt = 1.0
for i in 1:N_iter
    # Note that the parameters are exp-transformed when used as input
    # to SCAMPy
    @everywhere params_i = deepcopy(exp_transform(ekiobj.u[end]))
    @everywhere params_i = [params_i[i, :] for i in 1:size(params_i, 1)]
    g_ens_arr = pmap(g_, params_i)
    println("EKI iteration finished.")
    for j in 1:N_ens
      g_ens[j, :] = g_ens_arr[j]
    end
    EKI.update_ensemble!(ekiobj, g_ens; Δt=Δt)
    println("\nEnsemble covariance det. for iteration ", size(ekiobj.u)[1])
    println(det(cov(deepcopy((ekiobj.u[end])), dims=1)))
    # Save EKI information to file
    save("eki.jld", "eki_u", ekiobj.u, "eki_g", ekiobj.g,
        "truth_mean", ekiobj.g_t, "truth_cov", ekiobj.cov, "eki_err", ekiobj.err)
end

# EKI results: Has the ensemble collapsed toward the truth? Store and analyze.
println("\nEKI ensemble mean at last stage:")
println(mean(deepcopy(exp_transform(ekiobj.u[end])), dims=1))

println("\nEnsemble covariance det. 1st EKI, transformed space.")
println(det(cov(deepcopy((ekiobj.u[1])), dims=1)))

println("\nEnsemble covariance det. last EKI, transformed space.")
println(det(cov(deepcopy((ekiobj.u[end])), dims=1)))

# Save EKI information to file
save("eki.jld", "eki_u", ekiobj.u, "eki_g", ekiobj.g,
        "truth_mean", ekiobj.g_t, "truth_cov", ekiobj.cov, "eki_err", ekiobj.err)

###
###  Emulate: Gaussian Process Regression
###

gppackage = GPEmulator.GPJL()
pred_type = GPEmulator.YType()

mean_train_error = []
mean_test_error = []
for i in 1:N_iter
    # Get training and testing points:
    u_train, g_train, u_test, g_test = Utilities.extract_GP_train_test(ekiobj, i, 0.9)

    # Construct kernel:
    rbf_len = log.(ones(size(u_train, 2)))
    rbf_logstd = log(1.0)
    rbf = SEArd(rbf_len, rbf_logstd)
    # white noise for regularization. Right now it leads to convergence problems.
    # white = Noise(log(1.0e-4))
    # construct kernel
    GPkernel =  rbf

    normalized = true
    global gpobj = GPEmulator.GPObj(u_train, g_train, gppackage; GPkernel=GPkernel,
                             normalized=normalized, prediction_type=pred_type)

    # Check how well the Gaussian Process regression predicts on the
    # training and testing sets
    y_train_sqbias = GPEmulator.gp_sqbias(gpobj, u_train, g_train)
    println("Normalized mean training error:", mean(y_train_sqbias./yt_var) )
    append!(mean_train_error, mean(y_train_sqbias./yt_var))
    y_test_sqbias = GPEmulator.gp_sqbias(gpobj, u_test, g_test)
    println("Normalized mean testing error:", mean(y_test_sqbias./yt_var) )
    append!(mean_test_error, mean(y_test_sqbias./yt_var))

end
save("GP_errors.jld", "train_error", mean_train_error, "test_error", mean_test_error)


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
max_iter = 200000
burnin = 50000

# u0 has been modified by MCMCObj in the test, resetting.
u0 = vec(mean(ekiobj.u[end], dims=1))

mcmc = MCMC.MCMCObj(yt_sample, truth.cov, priors, 
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

plot(posterior[:, 1], posterior[:, 2], seriestype = :scatter)
savefig("MCMC_chain_transformed.png")

for idx in 1:n_param
    param = param_names_symb[idx]
    
    label = "true " * param
    histogram(posterior[:, idx], bins=100, normed=true, fill=:slategray, 
              lab="posterior", xlabel = param)
    xs = collect(logmeans[idx]-3.0*log_stds[idx]:log_stds[idx]/20.0:logmeans[idx]+3.0*log_stds[idx])
    plot!(xs, mcmc.prior[idx], w=2.6, color=:blue, lab="prior")

    title!(param)
    StatsPlots.savefig("posterior_"*param*".png")
end

# Plot the posteriors together with the priors and the 
#   true parameter values in the original space

plot(exp.(posterior[:, 1]), exp.(posterior[:, 2]), seriestype = :scatter, 
    xlabel = param_names_symb[1], ylabel=param_names_symb[2])
savefig("MCMC_chain_original.png")

histogram2d(exp.(posterior[:, 1]), exp.(posterior[:, 2]), 
    xlim = [0.0, 2.0], ylim = [0.0, 2.0], nbins=60, normed=true, 
    xlabel = param_names_symb[1], ylabel=param_names_symb[2], c= :bilbao)
savefig("Emulator_posterior_original.png")

cornerplot(exp.(posterior), compact = true, label = param_names_symb, histpct = 0.5, color = :rainbow)
savefig("cornerplot_original.png")

corrplot(exp.(posterior), label = param_names_symb)
savefig("corrplot_original.png")

marginalhist(exp.(posterior[:, 1]), exp.(posterior[:, 2]), 
    xlim = [0.0, 2.0], ylim = [0.0, 2.0], nbins=40, normed=true, 
    xguide = param_names_symb[1], yguide=param_names_symb[2], c= :bilbao)
savefig("marginalhist_original.png")

for idx in 1:n_param
    param = param_names_symb[idx]

    label = "true " * param
    histogram(exp.(posterior[:, idx]), bins=100, normed=true, fill=:slategray, 
              lab="posterior", xlabel = param)
    priors_ln = [Distributions.LogNormal(logmeans[1], log_stds[1]),
                        Distributions.LogNormal(logmeans[2], log_stds[2]),
                        Distributions.LogNormal(logmeans[3], log_stds[3])]

    xs = collect(logmeans[idx]-3.0*log_stds[idx]:log_stds[idx]/20.0:logmeans[idx]+3.0*log_stds[idx])
    plot!(exp.(xs), priors_ln[idx], w=2.6, color=:blue, lab="prior")

    title!(param)
    StatsPlots.savefig("posterior_orig_space_"*param*".png")
end
