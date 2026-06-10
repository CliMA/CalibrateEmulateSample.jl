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

EXPERIMENT = l96_experiment()  # respects EXPERIMENT env var / CLI arg

###########################################################################
#################### Metric parameters ###################################
###########################################################################

n_pushforward_samples         = 1000             # pushforward ensemble size for forcing/output metrics
n_lowrank_modes               = 5                # top EOF modes for perturbed-low-rank (aI+UDU') Mahalanobis
marginal_coverage_quantiles   = collect(0.05:0.05:0.95) # quantile levels for marginal coverage fraction
n_marginal_coverage_quantiles = length(marginal_coverage_quantiles)

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
y                      = prelim_data["y"]
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
forcing_samples_arr           = fill(NaN, n_rng, n_ens, n_k, n_pushforward_samples, n_forcing)
output_samples_arr            = fill(NaN, n_rng, n_ens, n_k, n_pushforward_samples, n_output)
forcing_mahal_arr             = fill(NaN, n_rng, n_ens, n_k)
forcing_logpdf_true_v_map_arr = fill(NaN, n_rng, n_ens, n_k)
output_mahal_arr              = fill(NaN, n_rng, n_ens, n_k)
output_logpdf_true_v_map_arr  = fill(NaN, n_rng, n_ens, n_k)
forcing_plr_mahal_top_arr      = fill(NaN, n_rng, n_ens, n_k)
forcing_plr_mahal_residual_arr = fill(NaN, n_rng, n_ens, n_k)
output_plr_mahal_top_arr       = fill(NaN, n_rng, n_ens, n_k)
output_plr_mahal_residual_arr  = fill(NaN, n_rng, n_ens, n_k)
forcing_coverage_arr = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)
output_coverage_arr  = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)
truth_forcing_arr    = fill(NaN, n_rng, n_ens, n_forcing)

for (N_ens, rng_idx) in valid_file_items
    post_fn = posterior_filename(cfg, N_ens, rng_idx)
    @info "loading case $(post_fn)"
    loaded = JLD2.load(joinpath(data_save_directory, post_fn))

    posteriors_by_k = loaded["posteriors_by_k"]
    k_values        = loaded["k_values"]
    truth_params             = loaded["truth_params"]
    truth_params_constrained = loaded["truth_params_constrained"]

    # Load ekpobj for total calibration cost
    ekp_loaded     = JLD2.load(joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx)))
    conv_alg_iters = length(get_g(ekp_loaded["ekpobj"]))

    i = findfirst(==(rng_idx), rng_idxs)
    j = findfirst(==(N_ens), N_enss)

    true_param_arr[i, j, :] = truth_params
    n_evals_arr[i, j]       = conv_alg_iters * N_ens

    emc_truth         = build_forcing(truth_phi, truth_params_constrained, phi_structure, sample_range)
    truth_forcing_vec = forcing(emc_truth, x0)
    truth_forcing_arr[i, j, :] = truth_forcing_vec

    for k in k_values
        post_dist = posteriors_by_k[k]
        pm = vec(mean(post_dist))
        pc = cov(post_dist)
        C_reg = Symmetric(pc + 1e-10 * I)
        post_normal  = MvNormal(pm, C_reg)
        post_samples = reduce(vcat, [get_distribution(post_dist)[name] for name in get_name(post_dist)])
        pmode        = post_samples[:, argmax(logpdf(post_normal, post_samples))]
        diff         = pm - truth_params

        num_samples = size(post_samples, 2)
        r = rank(pc)
        if r == num_samples - 1 && r < n_params - 1
            @warn "Posterior covariance rank $(r) = num_samples-1 = $(num_samples-1) < n_params-1 = $(n_params-1). Metric may be inaccurate due to insufficient samples; recommend num_samples > $(n_params)."
        end

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

        # forcing-space metrics (empirical distribution of pushforward samples vs truth forcing)
        fs = forcing_samples_arr[i, j, k, :, :]   # (n_pushforward_samples, n_forcing)
        fm = vec(mean(fs, dims=1))
        fc = Symmetric(cov(fs) + 1e-10 * I)
        f_normal = MvNormal(fm, fc)
        f_cols = Matrix(fs')
        f_mode = f_cols[:, argmax(logpdf(f_normal, f_cols))]
        f_diff = fm - truth_forcing_vec
        forcing_mahal_arr[i, j, k]             = f_diff' * (fc \ f_diff)
        forcing_logpdf_true_v_map_arr[i, j, k] = logpdf(f_normal, truth_forcing_vec) - logpdf(f_normal, f_mode)
        Ff  = eigen(fc)
        a_f = mean(Ff.values[1:end-n_lowrank_modes])
        V_f = Ff.vectors[:, end-n_lowrank_modes+1:end]
        λ_f = Ff.values[end-n_lowrank_modes+1:end]
        if a_f < 1e-8
            @warn "Forcing-space PLR skipped: noise floor a_f=$(a_f) ≈ regularization level (covariance near-degenerate, e.g. const-force). Entries left as NaN."
        else
            proj_f = V_f' * f_diff
            forcing_plr_mahal_top_arr[i, j, k]      = sum(proj_f.^2 ./ λ_f)
            forcing_plr_mahal_residual_arr[i, j, k] = (sum(f_diff.^2) - sum(proj_f.^2)) / a_f
        end
        for (qi, qp) in enumerate(marginal_coverage_quantiles)
            forcing_coverage_arr[i, j, k, qi] = mean(truth_forcing_vec .<= [quantile(fs[:, d], qp) for d in 1:n_forcing])
        end

        # output-space metrics (empirical distribution of pushforward samples vs truth output)
        os = output_samples_arr[i, j, k, :, :]    # (n_pushforward_samples, n_output)
        om = vec(mean(os, dims=1))
        oc = Symmetric(cov(os) + 1e-10 * I)
        o_normal = MvNormal(om, oc)
        o_cols = Matrix(os')
        o_mode = o_cols[:, argmax(logpdf(o_normal, o_cols))]
        o_diff = om - y
        output_mahal_arr[i, j, k]             = o_diff' * (oc \ o_diff)
        output_logpdf_true_v_map_arr[i, j, k] = logpdf(o_normal, y) - logpdf(o_normal, o_mode)
        Fo  = eigen(oc)
        a_o = mean(Fo.values[1:end-n_lowrank_modes])
        V_o = Fo.vectors[:, end-n_lowrank_modes+1:end]
        λ_o = Fo.values[end-n_lowrank_modes+1:end]
        if a_o < 1e-8
            @warn "Output-space PLR skipped: noise floor a_o=$(a_o) ≈ regularization level (covariance near-degenerate). Entries left as NaN."
        else
            proj_o = V_o' * o_diff
            output_plr_mahal_top_arr[i, j, k]      = sum(proj_o.^2 ./ λ_o)
            output_plr_mahal_residual_arr[i, j, k] = (sum(o_diff.^2) - sum(proj_o.^2)) / a_o
        end
        for (qi, qp) in enumerate(marginal_coverage_quantiles)
            output_coverage_arr[i, j, k, qi] = mean(y .<= [quantile(os[:, d], qp) for d in 1:n_output])
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
defDim(ds, "coverage_quantile", n_marginal_coverage_quantiles)

# Coordinate variables
rng_var = defVar(ds, "random_seed", Int64, ("random_seed",))
rng_var[:] = rng_idxs

ens_var = defVar(ds, "ensemble_size", Int64, ("ensemble_size",))
ens_var.attrib["description"] = "Number of ensemble members"
ens_var[:] = N_enss

k_var = defVar(ds, "k_iter", Int64, ("k_iter",))
k_var.attrib["description"] = "Number of EKP training iterations used to fit the emulator (1-indexed)"
k_var[:] = collect(1:n_k)

cov_q_var = defVar(ds, "coverage_quantile", Float64, ("coverage_quantile",))
cov_q_var.attrib["description"] = "Quantile levels used for marginal coverage fraction metrics"
cov_q_var[:] = marginal_coverage_quantiles

truth_forcing_v = defVar(ds, "truth_forcing", Float64, ("random_seed", "ensemble_size", "forcing_dim"); fillvalue=NaN)
truth_forcing_v.attrib["description"] = "True forcing vector for each (random_seed, ensemble_size) pair. Use with forcing_samples to recompute marginal coverage at arbitrary quantile levels."
truth_forcing_v[:, :, :] = truth_forcing_arr

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

forcing_mahal_v = defVar(
    ds, "forcing_mahalanobis", Float64,
    ("random_seed", "ensemble_size", "k_iter");
    fillvalue = NaN,
)
forcing_mahal_v.attrib["description"] = "Mahalanobis distance in forcing space: (fm - truth_forcing)' Cf^{-1} (fm - truth_forcing), where fm and Cf are the empirical mean/cov of the $(n_pushforward_samples) posterior pushforward forcing samples"
forcing_mahal_v[:, :, :] = forcing_mahal_arr

forcing_logpdf_true_v_map_v = defVar(
    ds, "forcing_logpdf_true_v_map", Float64,
    ("random_seed", "ensemble_size", "k_iter");
    fillvalue = NaN,
)
forcing_logpdf_true_v_map_v.attrib["description"] = "Log PDF ratio in forcing space: logpdf(N(fm,Cf), truth_forcing) - logpdf(N(fm,Cf), mode). Analogue of posterior_logpdf_true_v_map but for the pushforward forcing distribution"
forcing_logpdf_true_v_map_v[:, :, :] = forcing_logpdf_true_v_map_arr

output_mahal_v = defVar(
    ds, "output_mahalanobis", Float64,
    ("random_seed", "ensemble_size", "k_iter");
    fillvalue = NaN,
)
output_mahal_v.attrib["description"] = "Mahalanobis distance in output space: (om - y)' Co^{-1} (om - y), where om and Co are the empirical mean/cov of the $(n_pushforward_samples) posterior pushforward output samples, and y is the observations the posterior was fit to"
output_mahal_v[:, :, :] = output_mahal_arr

output_logpdf_true_v_map_v = defVar(
    ds, "output_logpdf_true_v_map", Float64,
    ("random_seed", "ensemble_size", "k_iter");
    fillvalue = NaN,
)
output_logpdf_true_v_map_v.attrib["description"] = "Log PDF ratio in output space: logpdf(N(om,Co), y) - logpdf(N(om,Co), mode). Analogue of posterior_logpdf_true_v_map but for the pushforward output distribution, with y the observations the posterior was fit to"
output_logpdf_true_v_map_v[:, :, :] = output_logpdf_true_v_map_arr

forcing_plr_top_v = defVar(ds, "forcing_plr_mahalanobis_top", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
forcing_plr_top_v.attrib["description"] = "Perturbed-low-rank Mahalanobis (top term) in forcing space: sum((Vf'*(fm-truth_forcing))^2 / λf_i), top $(n_lowrank_modes) eigenvectors/values of Cf ≈ a_f*I + Uf*Df*Uf'. Reference distribution: Chisq($(n_lowrank_modes)). Captures miscalibration in the k principal directions. Add forcing_plr_mahalanobis_residual for the full Chisq($(n_forcing)) statistic."
forcing_plr_top_v[:, :, :] = forcing_plr_mahal_top_arr

forcing_plr_res_v = defVar(ds, "forcing_plr_mahalanobis_residual", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
forcing_plr_res_v.attrib["description"] = "Perturbed-low-rank Mahalanobis (residual term) in forcing space: (||fm-truth_forcing||^2 - ||Vf'*(fm-truth_forcing)||^2) / a_f, where a_f = mean of the bottom $(n_forcing - n_lowrank_modes) eigenvalues of Cf. Reference distribution: Chisq($(n_forcing - n_lowrank_modes)). Captures miscalibration in the noise-floor directions. Add forcing_plr_mahalanobis_top for the full Chisq($(n_forcing)) statistic."
forcing_plr_res_v[:, :, :] = forcing_plr_mahal_residual_arr

output_plr_top_v = defVar(ds, "output_plr_mahalanobis_top", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
output_plr_top_v.attrib["description"] = "Perturbed-low-rank Mahalanobis (top term) in output space: sum((Vo'*(om-y))^2 / λo_i), top $(n_lowrank_modes) eigenvectors/values of Co ≈ a_o*I + Uo*Do*Uo'. Reference distribution: Chisq($(n_lowrank_modes)). Captures miscalibration in the k principal directions. Add output_plr_mahalanobis_residual for the full Chisq($(n_output)) statistic."
output_plr_top_v[:, :, :] = output_plr_mahal_top_arr

output_plr_res_v = defVar(ds, "output_plr_mahalanobis_residual", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
output_plr_res_v.attrib["description"] = "Perturbed-low-rank Mahalanobis (residual term) in output space: (||om-y||^2 - ||Vo'*(om-y)||^2) / a_o, where a_o = mean of the bottom $(n_output - n_lowrank_modes) eigenvalues of Co. Reference distribution: Chisq($(n_output - n_lowrank_modes)). Captures miscalibration in the noise-floor directions. Add output_plr_mahalanobis_top for the full Chisq($(n_output)) statistic."
output_plr_res_v[:, :, :] = output_plr_mahal_residual_arr

forcing_coverage_v = defVar(ds, "forcing_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue=NaN)
forcing_coverage_v.attrib["description"] = "Marginal coverage in forcing space: fraction of forcing dims where truth_forcing[d] ≤ q_p of the $(n_pushforward_samples) pushforward samples. Expected ≈ p for calibrated marginals."
forcing_coverage_v[:, :, :, :] = forcing_coverage_arr

output_coverage_v = defVar(ds, "output_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue=NaN)
output_coverage_v.attrib["description"] = "Marginal coverage in output space: fraction of output dims where y[d] ≤ q_p of the $(n_pushforward_samples) pushforward samples. Expected ≈ p for calibrated marginals."
output_coverage_v[:, :, :, :] = output_coverage_arr

close(ds)
@info "Saved leaderboard data to $(nc_save_filename)"
