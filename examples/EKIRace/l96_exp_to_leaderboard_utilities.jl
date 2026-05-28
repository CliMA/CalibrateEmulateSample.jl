using NCDatasets
using Dates
using JLD2
using Distributions
using LinearAlgebra
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses

# Step 1, Read and extract the available experiments
method_cases = ["Inversion", "TransformInversion", "Unscented", "GaussNewtonInversion"]
force_cases = ["const-force", "vec-force", "flux-force"] # problem types

#### CHOOSE YOUR CASE:
# calib_data_dir
method = method_cases[1]
calibrate_date = Date("2026-05-18", "yyyy-mm-dd")
calib_directory = "$(method)_$(calibrate_date)"

# calib_filename_suffix items to loop over
force_case = force_cases[2]
N_enss = [50,75,100] #[5, 15, 30]
rng_idxs = [1, 2, 3, 4]

### For saving the cases in the leaderboard style
forcing_cases_key = Dict(
    "const-force" => "ensemble_results",
    "vec-force" => "spatial_forcing_ensemble_results",
    "flux-force" => "nn_forcing_ensemble_results",
)

method_cases_key = Dict(
    "Inversion" => "ces-eki-dmc",
    "TransformInversion" => "ces-etki-dmc",
    "Unscented" => "ces-uki-dmc",
    "GaussNewtonInversion" => "ces-iekf",
)
nc_save_filename = "$(method_cases_key[method])_l96_$(forcing_cases_key[force_case])_$(calibrate_date).nc"

### determine valid files before we try to load

homedir = joinpath(pwd())
data_save_directory = joinpath(homedir, "output", calib_directory)

valid_file_items = []
valid_files = []
for N_ens in N_enss
    for rng_idx in rng_idxs
        calib_filename_suffix = "$(force_case)_$(N_ens)_$(rng_idx)"
        data_file = joinpath(data_save_directory, "posterior_$(calib_filename_suffix).jld2")
        if isfile(data_file)
            push!(valid_files, calib_filename_suffix)
            push!(valid_file_items, (force_case, N_ens, rng_idx))
        end
    end
end

@info "Converting data from valid files:"
display(valid_files)

if isempty(valid_file_items)
    error("No valid posterior files found in $(data_save_directory). Run emulate_sample_l96.jl first.")
end

### Load data

# Load first valid file to determine parameter dimension
first_loaded = JLD2.load(joinpath(data_save_directory, "posterior_$(valid_files[1]).jld2"))
n_params = length(vec(mean(first_loaded["posterior"])))

targets = [1.0]

n_rng     = length(rng_idxs)
n_ens     = length(N_enss)
n_targets = length(targets)

# Pre-allocate storage arrays indexed as (random_seed, ensemble_size, rmse_target, algorithm_type, ...)
true_param_arr = fill(NaN, n_rng, n_ens, n_targets, 1, n_params)
post_mean_arr  = fill(NaN, n_rng, n_ens, n_targets, 1, n_params)
post_cov_arr   = fill(NaN, n_rng, n_ens, n_targets, 1, n_params, n_params)
mahal_arr      = fill(NaN, n_rng, n_ens, n_targets, 1)
logpdf_true_v_map_arr     = fill(NaN, n_rng, n_ens, n_targets, 1)
n_evals_arr    = fill(NaN, n_rng, n_ens, n_targets, 1)

for (fc, N_ens, rng_idx) in valid_file_items
    calib_filename_suffix = "$(fc)_$(N_ens)_$(rng_idx)"
    post_filename = "posterior_$(calib_filename_suffix).jld2"
    @info "loading case $(post_filename)"
    loaded = JLD2.load(joinpath(data_save_directory, post_filename))
    
    post_dist    = loaded["posterior"]    # ParameterDistribution from MCMC chain (unconstrained space)
    truth_params = loaded["truth_params"] # truth in unconstrained space
    post_name = get_name(post_dist)
    
    pm = vec(mean(post_dist))  # posterior mean, shape (n_params,)
    pc = cov(post_dist)        # posterior covariance, shape (n_params, n_params)
    
    C_reg = Symmetric(pc + 1e-10 * I)
    post_normal = MvNormal(pm, C_reg)
    post_samples = reduce(vcat,[get_distribution(post_dist)[name] for name in get_name(post_dist)]) #samples are columns
    
    pmode_idx = argmax(logpdf(post_normal, post_samples))
    pmode = post_samples[:, pmode_idx]
    

    # Load ekpobj to compute calibration evaluation count
    ekp_loaded = JLD2.load(joinpath(data_save_directory, "l96_ekp_$(calib_filename_suffix).jld2"))
    conv_alg_iters = length(get_g(ekp_loaded["ekpobj"]))  # number of completed EKP update steps

    # Array indices
    i = findfirst(==(rng_idx), rng_idxs)
    j = findfirst(==(N_ens), N_enss)

    diff  = pm - truth_params
    
    for l in 1:n_targets
        true_param_arr[i, j, l, 1, :]  = truth_params
        post_mean_arr[i, j, l, 1, :]   = pm
        post_cov_arr[i, j, l, 1, :, :] = pc
        # (m - truth)' C^{-1} (m - truth)
        mahal_arr[i, j, l, 1]          = diff' * (C_reg \ diff)
        # log p(truth | posterior) - log p(mode | posterior)
        logpdf_true_v_map_arr[i, j, l, 1]         = logpdf(post_normal, truth_params) - logpdf(post_normal, pmode)
        # total forward model evaluations to reach target
        n_evals_arr[i, j, l, 1]        = conv_alg_iters * N_ens
    end

end


### Saving the data in nc

ds = NCDataset(nc_save_filename, "c")

defDim(ds, "random_seed",    n_rng)
defDim(ds, "ensemble_size",  n_ens)
defDim(ds, "rmse_target",    n_targets)
defDim(ds, "algorithm_type", 1)
defDim(ds, "param_dim",      n_params)
defDim(ds, "param_dim_2",    n_params)  # second param axis for covariance matrix

# Coordinate variables
alg_var = defVar(ds, "algorithm_type", String, ("algorithm_type",))
alg_var[1] = method_cases_key[method]

rng_var = defVar(ds, "random_seed", Int64, ("random_seed",))
rng_var[:] = rng_idxs

ens_var = defVar(ds, "ensemble_size", Int64, ("ensemble_size",))
ens_var.attrib["description"] = "Number of ensemble members"
ens_var[:] = N_enss

rmse_var = defVar(ds, "rmse_target", Float64, ("rmse_target",); fillvalue = NaN)
rmse_var.attrib["description"] = "Target accuracy level (root mean square error)"
rmse_var[:] = targets

# Data variables

n_evals_v = defVar(
    ds,
    "n_evals_to_target",
    Float64,
    ("random_seed", "ensemble_size", "rmse_target", "algorithm_type");
    fillvalue = NaN,
)
n_evals_v.attrib["description"] = "Number of forward model evaluations (i.e., algorithm cost): conv_alg_iters * ensemble_size. Stored as float in case of NaN values."
n_evals_v[:, :, :, :] = n_evals_arr

true_param_v = defVar(
    ds,
    "true_param",
    Float64,
    ("random_seed", "ensemble_size", "rmse_target", "algorithm_type", "param_dim");
    fillvalue = NaN,
)
true_param_v.attrib["description"] = "truth_param"
true_param_v[:, :, :, :, :] = true_param_arr

post_mean_v = defVar(
    ds,
    "post_mean",
    Float64,
    ("random_seed", "ensemble_size", "rmse_target", "algorithm_type", "param_dim");
    fillvalue = NaN,
)
post_mean_v.attrib["description"] = "mean(posterior)"
post_mean_v[:, :, :, :, :] = post_mean_arr

post_cov_v = defVar(
    ds,
    "post_cov",
    Float64,
    ("random_seed", "ensemble_size", "rmse_target", "algorithm_type", "param_dim", "param_dim_2");
    fillvalue = NaN,
)
post_cov_v.attrib["description"] = "cov(posterior)"
post_cov_v[:, :, :, :, :, :] = post_cov_arr

mahalanobis_v = defVar(
    ds,
    "mahalanobis",
    Float64,
    ("random_seed", "ensemble_size", "rmse_target", "algorithm_type");
    fillvalue = NaN,
)
mahalanobis_v.attrib["description"] = "M = mahalanobis distance: for posterior ≈ N(m,C), it is (m-truth_param)'*C^{-1}*(m-truth_param). Ideally M ∈ quantile(Chisq(dims), [0.05,0.95]) 90% of the time. It is the multidimensional equivalent of `how many standard deviations is the truth away form the posterior mean`. For gaussian -2P = M, if M > -2P => heavy tails"
mahalanobis_v[:, :, :, :] = mahal_arr

posterior_logpdf_true_v_map_v = defVar(
    ds,
    "posterior_logpdf_true_v_map",
    Float64,
    ("random_seed", "ensemble_size", "rmse_target", "algorithm_type");
    fillvalue = NaN,
)
posterior_logpdf_true_v_map_v.attrib["description"] = "P = posterior_logpdf(truth_param) - posterior_logpdf(mode(posterior)). Ideally -2P Ideally M ∈ quantile(Chisq(dims), [0.05,0.95]) 90% of the time. It asks `How much less probable is the true value compared to the MAP`. For gaussian -2P = M, if M > -2P => heavy tails"
posterior_logpdf_true_v_map_v[:, :, :, :] = logpdf_true_v_map_arr

close(ds)
@info "Saved leaderboard data to $(nc_save_filename)"
