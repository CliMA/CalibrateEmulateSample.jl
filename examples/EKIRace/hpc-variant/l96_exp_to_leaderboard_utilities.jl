using NCDatasets
using Dates
using JLD2
using Distributions
using LinearAlgebra
using Flux
using Random
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("experiment_config.jl")
include("Lorenz96.jl")

n_pushforward_samples = 1000

# Step 1, Read and extract the available experiments

#### CHOOSE YOUR CASE:
@assert EXPERIMENT in (:l96_const, :l96_vec, :l96_flux) "For l96_exp_to_leaderboard_utilities.jl, set EXPERIMENT to :l96_const, :l96_vec, or :l96_flux in experiment_config.jl"
cfg        = experiment_config(EXPERIMENT)
method     = method_cases[1]  # method_cases defined in experiment_config.jl
calib_dir  = calib_directory(method, cfg)
force_case = cfg.force_case
N_enss     = cfg.N_ens_sizes
rng_idxs   = collect(1:cfg.n_repeats)

nc_save_filename = nc_filename(cfg, method)

### determine valid files before we try to load

homedir = joinpath(pwd())
data_save_directory = joinpath(homedir, "output", calib_dir)

valid_file_items = []
valid_files = []
for N_ens in N_enss
    for rng_idx in rng_idxs
        data_file = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))
        if isfile(data_file)
            push!(valid_files, case_suffix(cfg, N_ens, rng_idx))
            push!(valid_file_items, (N_ens, rng_idx))
        end
    end
end

@info "Converting data from valid files:"
display(valid_files)

if isempty(valid_file_items)
    error("No valid posterior files found in $(data_save_directory). Run emulate_sample_l96.jl first.")
end

### Load data

# Load first valid file to determine parameter dimension and max k
first_loaded = JLD2.load(joinpath(data_save_directory, posterior_filename(cfg, valid_file_items[1]...)))
n_params = length(vec(mean(first_loaded["posteriors_by_k"][1])))

n_k = maximum(
    maximum(JLD2.load(joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx)))["k_values"])
    for (N_ens, rng_idx) in valid_file_items
)

n_rng = length(rng_idxs)
n_ens = length(N_enss)

# Load Lorenz preliminaries and truth structure for pushforward
prelim_file = joinpath(homedir, "output", "l96_computed_preliminaries_$(force_case).jld2")
isfile(prelim_file) || error("Prelim file not found at $(prelim_file). Run calibrate_l96.jl first.")
prelim_data            = JLD2.load(prelim_file)
x0                     = prelim_data["x0"]
nx                     = length(x0)
ic_cov_sqrt            = prelim_data["ic_cov_sqrt"]
lorenz_config_settings = prelim_data["lorenz_config_settings"]
observation_config     = prelim_data["observation_config"]
n_output               = 2 * nx

first_calib = JLD2.load(joinpath(data_save_directory, results_filename(cfg, valid_file_items[1]...)))
(truth_phi, _) = first_calib["truth_params_structure"]
(phi_structure, sample_range) = if force_case == "const-force"
    (nothing, nothing)
elseif force_case == "vec-force"
    (nothing, nothing)
elseif force_case == "flux-force"
    (truth_phi.model, truth_phi.sample_range)
end
n_forcing = force_case == "flux-force" ? length(sample_range) : nx

# Pre-allocate: posterior stats have a k_iter dimension; calibration cost and truth do not
true_param_arr        = fill(NaN, n_rng, n_ens, n_params)
n_evals_arr           = fill(NaN, n_rng, n_ens)
post_mean_arr         = fill(NaN, n_rng, n_ens, n_k, n_params)
post_cov_arr          = fill(NaN, n_rng, n_ens, n_k, n_params, n_params)
mahal_arr             = fill(NaN, n_rng, n_ens, n_k)
logpdf_true_v_map_arr = fill(NaN, n_rng, n_ens, n_k)
forcing_samples_arr   = fill(NaN, n_rng, n_ens, n_k, n_pushforward_samples, n_forcing)
output_samples_arr    = fill(NaN, n_rng, n_ens, n_k, n_pushforward_samples, n_output)

for (N_ens, rng_idx) in valid_file_items
    post_fn = posterior_filename(cfg, N_ens, rng_idx)
    @info "loading case $(post_fn)"
    loaded = JLD2.load(joinpath(data_save_directory, post_fn))

    posteriors_by_k = loaded["posteriors_by_k"]
    k_values        = loaded["k_values"]
    truth_params    = loaded["truth_params"]

    # Load ekpobj for total calibration cost
    ekp_loaded     = JLD2.load(joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx)))
    conv_alg_iters = length(get_g(ekp_loaded["ekpobj"]))

    i = findfirst(==(rng_idx), rng_idxs)
    j = findfirst(==(N_ens), N_enss)

    true_param_arr[i, j, :] = truth_params
    n_evals_arr[i, j]       = conv_alg_iters * N_ens

    for k in k_values
        post_dist = posteriors_by_k[k]
        pm = vec(mean(post_dist))
        pc = cov(post_dist)
        C_reg = Symmetric(pc + 1e-10 * I)
        post_normal  = MvNormal(pm, C_reg)
        post_samples = reduce(vcat, [get_distribution(post_dist)[name] for name in get_name(post_dist)])
        pmode        = post_samples[:, argmax(logpdf(post_normal, post_samples))]
        diff         = pm - truth_params

        post_mean_arr[i, j, k, :]   = pm
        post_cov_arr[i, j, k, :, :] = pc
        # (m - truth)' C^{-1} (m - truth)
        mahal_arr[i, j, k]           = diff' * (C_reg \ diff)
        # log p(truth | posterior) - log p(mode | posterior)
        logpdf_true_v_map_arr[i, j, k] = logpdf(post_normal, truth_params) - logpdf(post_normal, pmode)

        # pushforward ensemble for forcing and output space
        push_unc = sample(post_dist, n_pushforward_samples)
        push_con = transform_unconstrained_to_constrained(post_dist, push_unc)
        @info "Pushforward k=$(k), N_ens=$(N_ens), rng_idx=$(rng_idx): $(n_pushforward_samples) Lorenz forward maps"
        for s in 1:n_pushforward_samples
            emc = build_forcing(truth_phi, push_con[:, s], phi_structure, sample_range)
            forcing_samples_arr[i, j, k, s, :] = forcing(emc, x0)
            output_samples_arr[i, j, k, s, :]  = lorenz_forward(
                emc,
                x0 .+ ic_cov_sqrt * rand(Normal(0.0, 1.0), nx, 1),
                lorenz_config_settings,
                observation_config,
            )
        end
    end
end


### Saving the data in nc

ds = NCDataset(nc_save_filename, "c")

defDim(ds, "random_seed",       n_rng)
defDim(ds, "ensemble_size",     n_ens)
defDim(ds, "k_iter",            n_k)
defDim(ds, "param_dim",         n_params)
defDim(ds, "param_dim_2",       n_params)
defDim(ds, "pushforward_sample", n_pushforward_samples)
defDim(ds, "forcing_dim",       n_forcing)
defDim(ds, "output_dim",        n_output)

# Coordinate variables
rng_var = defVar(ds, "random_seed", Int64, ("random_seed",))
rng_var[:] = rng_idxs

ens_var = defVar(ds, "ensemble_size", Int64, ("ensemble_size",))
ens_var.attrib["description"] = "Number of ensemble members"
ens_var[:] = N_enss

k_var = defVar(ds, "k_iter", Int64, ("k_iter",))
k_var.attrib["description"] = "Number of EKP training iterations used to fit the emulator (1-indexed)"
k_var[:] = collect(1:n_k)

# Data variables

n_evals_v = defVar(
    ds, "n_evals_to_target", Float64,
    ("random_seed", "ensemble_size");
    fillvalue = NaN,
)
n_evals_v.attrib["description"] = "Total calibration cost: conv_alg_iters * ensemble_size."
n_evals_v[:, :] = n_evals_arr

true_param_v = defVar(
    ds, "true_param", Float64,
    ("random_seed", "ensemble_size", "param_dim");
    fillvalue = NaN,
)
true_param_v.attrib["description"] = "truth_param"
true_param_v[:, :, :] = true_param_arr

post_mean_v = defVar(
    ds, "post_mean", Float64,
    ("random_seed", "ensemble_size", "k_iter", "param_dim");
    fillvalue = NaN,
)
post_mean_v.attrib["description"] = "mean(posterior) for each emulator training size k"
post_mean_v[:, :, :, :] = post_mean_arr

post_cov_v = defVar(
    ds, "post_cov", Float64,
    ("random_seed", "ensemble_size", "k_iter", "param_dim", "param_dim_2");
    fillvalue = NaN,
)
post_cov_v.attrib["description"] = "cov(posterior) for each emulator training size k"
post_cov_v[:, :, :, :, :] = post_cov_arr

mahalanobis_v = defVar(
    ds, "mahalanobis", Float64,
    ("random_seed", "ensemble_size", "k_iter");
    fillvalue = NaN,
)
mahalanobis_v.attrib["description"] = "M = mahalanobis distance: for posterior ≈ N(m,C), it is (m-truth_param)'*C^{-1}*(m-truth_param). Ideally M ∈ quantile(Chisq(dims), [0.05,0.95]) 90% of the time. It is the multidimensional equivalent of `how many standard deviations is the truth away form the posterior mean`. For gaussian -2P = M, if M > -2P => heavy tails"
mahalanobis_v[:, :, :] = mahal_arr

posterior_logpdf_true_v_map_v = defVar(
    ds, "posterior_logpdf_true_v_map", Float64,
    ("random_seed", "ensemble_size", "k_iter");
    fillvalue = NaN,
)
posterior_logpdf_true_v_map_v.attrib["description"] = "P = posterior_logpdf(truth_param) - posterior_logpdf(mode(posterior)). Ideally -2P Ideally M ∈ quantile(Chisq(dims), [0.05,0.95]) 90% of the time. It asks `How much less probable is the true value compared to the MAP`. For gaussian -2P = M, if M > -2P => heavy tails"
posterior_logpdf_true_v_map_v[:, :, :] = logpdf_true_v_map_arr

forcing_samples_v = defVar(
    ds, "forcing_samples", Float64,
    ("random_seed", "ensemble_size", "k_iter", "pushforward_sample", "forcing_dim");
    fillvalue = NaN,
)
forcing_samples_v.attrib["description"] = "$(n_pushforward_samples) posterior pushforward samples mapped to forcing space"
forcing_samples_v[:, :, :, :, :] = forcing_samples_arr

output_samples_v = defVar(
    ds, "output_samples", Float64,
    ("random_seed", "ensemble_size", "k_iter", "pushforward_sample", "output_dim");
    fillvalue = NaN,
)
output_samples_v.attrib["description"] = "$(n_pushforward_samples) posterior pushforward samples mapped to Lorenz96 output space (mean/std per spatial point)"
output_samples_v[:, :, :, :, :] = output_samples_arr

close(ds)
@info "Saved leaderboard data to $(nc_save_filename)"
