# Import modules
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using Distributions  # probability distributions and associated functions
@everywhere using StatsBase
@everywhere using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
@everywhere using CalibrateEmulateSample.EKI
@everywhere using CalibrateEmulateSample.Observations
@everywhere using CalibrateEmulateSample.Utilities
@everywhere using CalibrateEmulateSample.Transformations
@everywhere using CalibrateEmulateSample.SCModel

using JLD

###
###  Define the parameters and their priors
###

# Define the parameters that we want to learn

@everywhere param_names = ["entrainment_factor", "detrainment_factor", "sorting_power", 
	"tke_ed_coeff", "tke_diss_coeff", "pressure_normalmode_adv_coeff", 
         "pressure_normalmode_buoy_coeff1", "pressure_normalmode_drag_coeff", "static_stab_coeff"]
@everywhere n_param = length(param_names)

# Assume lognormal priors for all parameters
# Note: For the EDMF model to run, all parameters need to be nonnegative. 
# The EKI update can result in violations of 
# these constraints - therefore, we perform CES in log space, i.e.,
# (the parameters can then simply be obtained by exponentiating the final results). 

# Prior option 1: Log-normal in original space defined by mean and std
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
priors = [Distributions.Normal(logmeans[1], log_stds[1]),
                        Distributions.Normal(logmeans[2], log_stds[2]),
                        Distributions.Normal(logmeans[3], log_stds[3]),
                        Distributions.Normal(logmeans[4], log_stds[4]),
                        Distributions.Normal(logmeans[5], log_stds[5]),
                        Distributions.Normal(logmeans[6], log_stds[6]),
                        Distributions.Normal(logmeans[7], log_stds[7]),
                        Distributions.Normal(logmeans[8], log_stds[8]),
                        Distributions.Normal(logmeans[9], log_stds[9])]
@everywhere priors = $priors
# Second option: Uniform distributions
#@everywhere priors = [Distributions.Uniform(0, 1),
#                        Distributions.Uniform(0, 1),
#                        Distributions.Uniform(0, 4),
#                        Distributions.Uniform(0, 1),
#                        Distributions.Uniform(0, 1),
#                        Distributions.Uniform(0, 1),
#                        Distributions.Uniform(0, 1),
#                        Distributions.Uniform(3, 10),
#                        Distributions.Uniform(0, 1) ]


# # Prior option 2: Log-normal in original space defined by mode and std
# logmean_c_m, logstd_c_m = logmean_and_logstd_from_mode_std(0.5, 0.3)
# logmean_c_ϵ, logstd_c_ϵ = logmean_and_logstd_from_mode_std(0.5, 0.3)
# println("Logmean and logstd of the prior: ", logmean_c_m, " ", logstd_c_m)
# priors = [Distributions.Normal(logmean_c_m, logstd_c_m),    # prior on c_m
#           Distributions.Normal(logmean_c_ϵ, logstd_c_ϵ)]    # prior on c_ϵ

###
###  Retrieve true LES samples
###

# This is the true value of the observables (e.g. LES ensemble mean for EDMF)
@everywhere ti = [10800.0, 28800.0, 10800.0, 18000.0]
@everywhere tf = [14400.0, 32400.0, 14400.0, 21600.0]
y_names = Array{String, 1}[]
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #DYCOMS_RF01
push!(y_names, ["thetal_mean", "u_mean", "v_mean", "tke_mean"]) #GABLS
push!(y_names, ["thetal_mean", "total_flux_h"]) #Nieuwstadt
push!(y_names, ["thetal_mean", "ql_mean", "qt_mean", "total_flux_h", "total_flux_qt"]) #Bomex
@everywhere y_names=$y_names

# Get observations
@everywhere yt = zeros(0)
@everywhere yt_var = zeros(0)
@everywhere sim_names = ["DYCOMS_RF01", "GABLS", "Nieuwstadt", "Bomex"]

les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[1],".may20")
sim_dir = string("Output.", sim_names[1],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[1], les_dir, ti[1], tf[1], z_scm = z_scm)
append!(yt, yt_)
append!(yt_var, yt_var_)

les_dir = string("/groups/esm/ilopezgo/Output.", sim_names[2],".iles128wCov")
sim_dir = string("Output.", sim_names[2],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[2], les_dir, ti[2], tf[2], z_scm = z_scm)
append!(yt, yt_)
append!(yt_var, yt_var_)

les_dir = string("/groups/esm/ilopezgo/Output.Soares.dry11")
sim_dir = string("Output.", sim_names[3],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[3], les_dir, ti[3], tf[3], z_scm = z_scm)
append!(yt, yt_)
append!(yt_var, yt_var_)

les_dir = string("/groups/esm/ilopezgo/Output.Bomex.may18")
sim_dir = string("Output.", sim_names[4],".00000")
z_scm = get_profile(sim_dir, ["z_half"])
yt_, yt_var_ = obs_LES(y_names[4], les_dir, ti[4], tf[4], z_scm = z_scm)
append!(yt, yt_)
append!(yt_var, yt_var_)

@everywhere yt = $yt
@everywhere yt_var = $yt_var
@everywhere n_observables = length(yt)

# This is how many samples of the true data we have
n_samples = 1
samples = zeros(n_samples, length(yt))
samples[1,:] = yt
# Noise level of the samples, which scales the time variance of each output.
noise_level = 1.0
Γy = noise_level^2 * convert(Array, Diagonal(yt_var))
μ_noise = zeros(length(yt))

# We construct the observations object with the samples and the cov.
truth = Observations.Obs(samples, Γy, y_names[1])
@everywhere truth = $truth

###
###  Calibrate: Ensemble Kalman Inversion
###

@everywhere N_ens = 5 # number of ensemble members
@everywhere N_iter = 1 # number of EKI iterations.
@everywhere N_yt = length(yt) # Length of data array

@everywhere initial_params = EKI.construct_initial_ensemble(N_ens, priors)
precondition_ensemble!(initial_params, priors, param_names, y_names, ti, tf)
@everywhere initial_params = $initial_params

@everywhere ekiobj = EKI.EKIObj(initial_params, param_names, truth.mean, truth.obs_noise_cov)
g_ens = zeros(N_ens, n_observables)

@everywhere scm_dir = "/home/ilopezgo/SCAMPy/"
@everywhere params_i = deepcopy(exp_transform(ekiobj.u[end]))

@everywhere g_(x::Array{Float64,1}) = run_SCAMPy(x, param_names,
   y_names, scm_dir, ti, tf)

outdir_path = string("results_p", n_param,"_n", noise_level,"_e", N_ens, "_i", N_iter, "_d", N_yt)
command = `mkdir $outdir_path`
try
    run(command)
catch e
    println("Output directory already exists. Output may be overwritten.")
end

# EKI iterations
@everywhere Δt = 1.0
for i in 1:N_iter
    # Note that the parameters are exp-transformed when used as input
    # to SCAMPy
    @everywhere params_i = deepcopy(exp_transform(ekiobj.u[end]))
    @everywhere params_i = [params_i[i, :] for i in 1:size(params_i, 1)]
    g_ens_arr = pmap(g_, params_i)
    println(string("\n\nEKI evaluation ",i," finished. Updating ensemble ...\n"))
    for j in 1:N_ens
      g_ens[j, :] = g_ens_arr[j]
    end
    EKI.update_ensemble!(ekiobj, g_ens; Δt=Δt)
    println("\nEnsemble updated.\n")
    println("\nEnsemble covariance det. for iteration ", size(ekiobj.u)[1])
    println(det(cov(deepcopy((ekiobj.u[end])), dims=1)))
    # Save EKI information to file
    save( string(outdir_path,"/eki.jld"), "eki_u", ekiobj.u, "eki_g", ekiobj.g,
        "truth_mean", ekiobj.g_t, "truth_cov", ekiobj.cov, "eki_err", ekiobj.err)
end

# Save EKI information to file
save("eki.jld", "eki_u", ekiobj.u, "eki_g", ekiobj.g,
        "truth_mean", ekiobj.g_t, "truth_cov", ekiobj.cov, "eki_err", ekiobj.err)

# EKI results: Has the ensemble collapsed toward the truth? Store and analyze.
println("\nEKI ensemble mean at last stage (original space):")
println(mean(deepcopy(exp_transform(ekiobj.u[end])), dims=1))

println("\nEnsemble covariance det. 1st iteration, transformed space.")
println(det(cov(deepcopy((ekiobj.u[1])), dims=1)))

println("\nEnsemble covariance det. last iteration, transformed space.")
println(det(cov(deepcopy((ekiobj.u[end])), dims=1)))
