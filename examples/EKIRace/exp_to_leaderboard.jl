using NCDatasets
using Dates
using JLD2
using Distributions
using LinearAlgebra
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("experiment_config.jl")

###########################################################################
#################### Metric parameters ###################################
###########################################################################

n_lowrank_modes               = 5
marginal_coverage_quantiles   = collect(0.05:0.05:0.95)
n_marginal_coverage_quantiles = length(marginal_coverage_quantiles)
budget_target_scalings        = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
n_target_scalings             = length(budget_target_scalings)

###########################################################################
#################### Config setup ########################################
###########################################################################

cfg         = experiment_config(EXPERIMENT)
method      = method_cases[1]
calib_dir   = calib_directory(method, cfg)
N_enss      = cfg.N_ens_sizes
rng_idxs    = collect(1:cfg.n_repeats)
has_forcing = cfg.force_case !== nothing

nc_save_filename = nc_filename(cfg, method)

###########################################################################
#################### Locate valid posterior files ########################
###########################################################################

homedir             = joinpath(pwd())
data_save_directory = joinpath(homedir, "output", calib_dir)

valid_file_items = []
valid_files      = []
for N_ens in N_enss, rng_idx in rng_idxs
    data_file = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))
    if isfile(data_file)
        push!(valid_files, case_suffix(cfg, N_ens, rng_idx))
        push!(valid_file_items, (N_ens, rng_idx))
    end
end

@info "Converting data from valid files:"
display(valid_files)

if isempty(valid_file_items)
    error("No valid posterior files found in $(data_save_directory). Run emulate_sample first.")
end

###########################################################################
#################### Determine dimensions ################################
###########################################################################

first_post_fn = joinpath(data_save_directory, posterior_filename(cfg, valid_file_items[1]...))
first_loaded  = JLD2.load(first_post_fn)
if !haskey(first_loaded, "pushforward_output_samples")
    error("Pushforward data not found in $(first_post_fn). Run pushforward_from_posterior_$(cfg.model).jl first.")
end

n_params              = length(vec(mean(first_loaded["posteriors_by_k"][1])))
n_pushforward_samples = first_loaded["pushforward_n_samples"]
n_output              = size(first_loaded["pushforward_output_samples"], 2)
n_forcing             = has_forcing ? size(first_loaded["pushforward_forcing_samples"], 2) : 0

n_k = maximum(
    maximum(JLD2.load(joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx)))["k_values"])
    for (N_ens, rng_idx) in valid_file_items
)

n_rng = length(rng_idxs)
n_ens = length(N_enss)

###########################################################################
#################### Load truth observation vector #######################
###########################################################################

prelim_file = if cfg.force_case === nothing
    joinpath(homedir, "output", "$(cfg.model)_computed_preliminaries.jld2")
else
    joinpath(homedir, "output", "$(cfg.model)_computed_preliminaries_$(cfg.force_case).jld2")
end
isfile(prelim_file) || error("Prelim file not found at $(prelim_file). Run calibrate first.")
y = JLD2.load(prelim_file, "y")

###########################################################################
#################### Pre-allocate arrays #################################
###########################################################################

true_param_arr        = fill(NaN, n_rng, n_ens, n_params)
n_evals_arr           = fill(NaN, n_rng, n_ens)
post_mean_arr         = fill(NaN, n_rng, n_ens, n_k, n_params)
post_cov_arr          = fill(NaN, n_rng, n_ens, n_k, n_params, n_params)
mahal_arr             = fill(NaN, n_rng, n_ens, n_k)
logpdf_true_v_map_arr = fill(NaN, n_rng, n_ens, n_k)
output_samples_arr            = fill(NaN, n_rng, n_ens, n_k, n_pushforward_samples, n_output)
output_mahal_arr              = fill(NaN, n_rng, n_ens, n_k)
output_logpdf_true_v_map_arr  = fill(NaN, n_rng, n_ens, n_k)
output_plr_mahal_top_arr      = fill(NaN, n_rng, n_ens, n_k)
output_plr_mahal_residual_arr = fill(NaN, n_rng, n_ens, n_k)
output_coverage_arr           = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)

if has_forcing
    forcing_samples_arr           = fill(NaN, n_rng, n_ens, n_k, n_pushforward_samples, n_forcing)
    forcing_mahal_arr             = fill(NaN, n_rng, n_ens, n_k)
    forcing_logpdf_true_v_map_arr = fill(NaN, n_rng, n_ens, n_k)
    forcing_plr_mahal_top_arr     = fill(NaN, n_rng, n_ens, n_k)
    forcing_plr_mahal_residual_arr = fill(NaN, n_rng, n_ens, n_k)
    forcing_coverage_arr          = fill(NaN, n_rng, n_ens, n_k, n_marginal_coverage_quantiles)
    truth_forcing_arr             = fill(NaN, n_rng, n_ens, n_forcing)
end

###########################################################################
#################### Main loop over posterior files ######################
###########################################################################

for (N_ens, rng_idx) in valid_file_items
    post_fn = posterior_filename(cfg, N_ens, rng_idx)
    @info "loading case $(post_fn)"
    loaded = JLD2.load(joinpath(data_save_directory, post_fn))

    if !haskey(loaded, "pushforward_output_samples")
        @warn "Pushforward data missing for $(post_fn); skipping. Run pushforward_from_posterior_$(cfg.model).jl first."
        continue
    end

    posteriors_by_k = loaded["posteriors_by_k"]
    k_values        = loaded["k_values"]
    truth_params    = loaded["truth_params"]

    pf_output   = loaded["pushforward_output_samples"]   # (n_samples, n_output, n_k_pos)
    pf_k_values = loaded["pushforward_k_values"]

    if has_forcing
        pf_forcing        = loaded["pushforward_forcing_samples"]  # (n_samples, n_forcing, n_k_pos)
        truth_forcing_vec = loaded["truth_forcing"]
    end

    ekp_loaded     = JLD2.load(joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx)))
    conv_alg_iters = length(get_g(ekp_loaded["ekpobj"]))

    i = findfirst(==(rng_idx), rng_idxs)
    j = findfirst(==(N_ens), N_enss)

    true_param_arr[i, j, :] = truth_params
    n_evals_arr[i, j]       = conv_alg_iters * N_ens
    if has_forcing
        truth_forcing_arr[i, j, :] = truth_forcing_vec
    end

    for k in k_values
        post_dist = posteriors_by_k[k]
        pm = vec(mean(post_dist))
        pc = cov(post_dist)
        C_reg        = Symmetric(pc + 1e-10 * I)
        post_normal  = MvNormal(pm, C_reg)
        post_samples = reduce(vcat, [get_distribution(post_dist)[name] for name in get_name(post_dist)])
        pmode        = post_samples[:, argmax(logpdf(post_normal, post_samples))]
        diff         = pm - truth_params

        num_samples = size(post_samples, 2)
        r = rank(pc)
        if r == num_samples - 1 && r < n_params - 1
            @warn "Posterior covariance rank $(r) = num_samples-1 = $(num_samples-1) < n_params-1 = $(n_params-1). Metric may be inaccurate; recommend num_samples > $(n_params)."
        end

        post_mean_arr[i, j, k, :]      = pm
        post_cov_arr[i, j, k, :, :]    = pc
        mahal_arr[i, j, k]             = diff' * (C_reg \ diff)
        logpdf_true_v_map_arr[i, j, k] = logpdf(post_normal, truth_params) - logpdf(post_normal, pmode)

        ki = findfirst(==(k), pf_k_values)

        # ── Output-space metrics ──────────────────────────────────────────
        output_samples_arr[i, j, k, :, :] = pf_output[:, :, ki]
        os = output_samples_arr[i, j, k, :, :]
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
            @warn "Output-space PLR skipped: noise floor a_o=$(a_o) ≈ regularization level. Entries left as NaN."
        else
            proj_o = V_o' * o_diff
            output_plr_mahal_top_arr[i, j, k]      = sum(proj_o.^2 ./ λ_o)
            output_plr_mahal_residual_arr[i, j, k] = (sum(o_diff.^2) - sum(proj_o.^2)) / a_o
        end
        for (qi, qp) in enumerate(marginal_coverage_quantiles)
            output_coverage_arr[i, j, k, qi] = mean(y .<= [quantile(os[:, d], qp) for d in 1:n_output])
        end

        # ── Forcing-space metrics (L96 only) ─────────────────────────────
        if has_forcing
            forcing_samples_arr[i, j, k, :, :] = pf_forcing[:, :, ki]
            fs = forcing_samples_arr[i, j, k, :, :]
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
                @warn "Forcing-space PLR skipped: noise floor a_f=$(a_f) ≈ regularization level (e.g. const-force). Entries left as NaN."
            else
                proj_f = V_f' * f_diff
                forcing_plr_mahal_top_arr[i, j, k]      = sum(proj_f.^2 ./ λ_f)
                forcing_plr_mahal_residual_arr[i, j, k] = (sum(f_diff.^2) - sum(proj_f.^2)) / a_f
            end
            for (qi, qp) in enumerate(marginal_coverage_quantiles)
                forcing_coverage_arr[i, j, k, qi] = mean(truth_forcing_vec .<= [quantile(fs[:, d], qp) for d in 1:n_forcing])
            end
        end
    end
end

###########################################################################
#################### Budget for coverage #################################
###########################################################################

output_budget_to_target = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
output_iters_to_target  = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
for (si, c) in enumerate(budget_target_scalings)
    tol = c .* sqrt.(marginal_coverage_quantiles .* (1 .- marginal_coverage_quantiles) ./ n_output)
    for ri in 1:n_rng, ei in 1:n_ens, (qi, qp) in enumerate(marginal_coverage_quantiles)
        for k in 1:n_k
            s = output_coverage_arr[ri, ei, k, qi]
            isnan(s) && continue
            if abs(s - qp) <= tol[qi]
                output_budget_to_target[ri, ei, si, qi] = N_enss[ei] * k
                output_iters_to_target[ri, ei, si, qi]  = k
                break
            end
        end
    end
end

if has_forcing
    forcing_budget_to_target = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    forcing_iters_to_target  = fill(NaN, n_rng, n_ens, n_target_scalings, n_marginal_coverage_quantiles)
    for (si, c) in enumerate(budget_target_scalings)
        tol = c .* sqrt.(marginal_coverage_quantiles .* (1 .- marginal_coverage_quantiles) ./ n_forcing)
        for ri in 1:n_rng, ei in 1:n_ens, (qi, qp) in enumerate(marginal_coverage_quantiles)
            for k in 1:n_k
                s = forcing_coverage_arr[ri, ei, k, qi]
                isnan(s) && continue
                if abs(s - qp) <= tol[qi]
                    forcing_budget_to_target[ri, ei, si, qi] = N_enss[ei] * k
                    forcing_iters_to_target[ri, ei, si, qi]  = k
                    break
                end
            end
        end
    end
end

###########################################################################
#################### Save to NetCDF #####################################
###########################################################################

ds = NCDataset(nc_save_filename, "c")

defDim(ds, "random_seed",        n_rng)
defDim(ds, "ensemble_size",      n_ens)
defDim(ds, "k_iter",             n_k)
defDim(ds, "param_dim",          n_params)
defDim(ds, "param_dim_2",        n_params)
defDim(ds, "pushforward_sample", n_pushforward_samples)
defDim(ds, "output_dim",         n_output)
defDim(ds, "coverage_quantile",  n_marginal_coverage_quantiles)
defDim(ds, "target_scaling",     n_target_scalings)
if has_forcing
    defDim(ds, "forcing_dim", n_forcing)
end

# ── Coordinate variables ──────────────────────────────────────────────
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

ts_var = defVar(ds, "target_scaling", Float64, ("target_scaling",))
ts_var.attrib["description"] = "Scaling c in α_c(q) = c·√(q(1−q)/N_y); tolerance for budget_to_target / iters_to_target"
ts_var[:] = budget_target_scalings

# ── Calibration cost and truth ────────────────────────────────────────
n_evals_v = defVar(ds, "n_evals_to_target", Float64, ("random_seed", "ensemble_size"); fillvalue=NaN)
n_evals_v.attrib["description"] = "Total calibration cost: conv_alg_iters * ensemble_size."
n_evals_v[:, :] = n_evals_arr

true_param_v = defVar(ds, "true_param", Float64, ("random_seed", "ensemble_size", "param_dim"); fillvalue=NaN)
true_param_v.attrib["description"] = "truth_param"
true_param_v[:, :, :] = true_param_arr

if has_forcing
    truth_forcing_v = defVar(ds, "truth_forcing", Float64, ("random_seed", "ensemble_size", "forcing_dim"); fillvalue=NaN)
    truth_forcing_v.attrib["description"] = "True forcing vector for each (random_seed, ensemble_size) pair."
    truth_forcing_v[:, :, :] = truth_forcing_arr
end

# ── Param-space metrics ───────────────────────────────────────────────
post_mean_v = defVar(ds, "post_mean", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim"); fillvalue=NaN)
post_mean_v.attrib["description"] = "mean(posterior) for each emulator training size k"
post_mean_v[:, :, :, :] = post_mean_arr

post_cov_v = defVar(ds, "post_cov", Float64, ("random_seed", "ensemble_size", "k_iter", "param_dim", "param_dim_2"); fillvalue=NaN)
post_cov_v.attrib["description"] = "cov(posterior) for each emulator training size k"
post_cov_v[:, :, :, :, :] = post_cov_arr

mahalanobis_v = defVar(ds, "mahalanobis", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
mahalanobis_v.attrib["description"] = "M = (m-truth)' C^{-1} (m-truth) in parameter space."
mahalanobis_v[:, :, :] = mahal_arr

posterior_logpdf_true_v_map_v = defVar(ds, "posterior_logpdf_true_v_map", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
posterior_logpdf_true_v_map_v.attrib["description"] = "P = logpdf(posterior, truth_param) - logpdf(posterior, mode). For Gaussian, -2P = M."
posterior_logpdf_true_v_map_v[:, :, :] = logpdf_true_v_map_arr

# ── Output-space metrics ──────────────────────────────────────────────
output_samples_v = defVar(ds, "output_samples", Float64, ("random_seed", "ensemble_size", "k_iter", "pushforward_sample", "output_dim"); fillvalue=NaN)
output_samples_v.attrib["description"] = "$(n_pushforward_samples) posterior pushforward samples in output space"
output_samples_v[:, :, :, :, :] = output_samples_arr

output_mahal_v = defVar(ds, "output_mahalanobis", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
output_mahal_v.attrib["description"] = "Mahalanobis distance in output space: (om-y)' Co^{-1} (om-y)."
output_mahal_v[:, :, :] = output_mahal_arr

output_logpdf_true_v_map_v = defVar(ds, "output_logpdf_true_v_map", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
output_logpdf_true_v_map_v.attrib["description"] = "Log PDF ratio in output space: logpdf(N(om,Co), y) - logpdf(N(om,Co), mode)."
output_logpdf_true_v_map_v[:, :, :] = output_logpdf_true_v_map_arr

output_plr_top_v = defVar(ds, "output_plr_mahalanobis_top", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
output_plr_top_v.attrib["description"] = "PLR Mahalanobis top term in output space: sum((Vo'*(om-y))^2 / λo_i), top $(n_lowrank_modes) modes. Ref: Chisq($(n_lowrank_modes))."
output_plr_top_v[:, :, :] = output_plr_mahal_top_arr

output_plr_res_v = defVar(ds, "output_plr_mahalanobis_residual", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
output_plr_res_v.attrib["description"] = "PLR Mahalanobis residual term in output space: (||om-y||^2 - ||Vo'*(om-y)||^2) / a_o. Ref: Chisq($(n_output - n_lowrank_modes))."
output_plr_res_v[:, :, :] = output_plr_mahal_residual_arr

output_coverage_v = defVar(ds, "output_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue=NaN)
output_coverage_v.attrib["description"] = "Marginal coverage in output space: fraction of output dims where y[d] ≤ q_p of the $(n_pushforward_samples) pushforward samples."
output_coverage_v[:, :, :, :] = output_coverage_arr

output_budget_v = defVar(ds, "output_budget_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue=NaN)
output_budget_v.attrib["description"] = "Budget (N_ens·k_iter) to first reach |S(q)−q| ≤ c·√(q(1−q)/N_y) per quantile q in output space. NaN = not reached. Take max over quantiles for the all-quantile condition."
output_budget_v[:, :, :, :] = output_budget_to_target

output_iters_v = defVar(ds, "output_iters_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue=NaN)
output_iters_v.attrib["description"] = "Iterations k_iter to first reach coverage target per quantile in output space. NaN = not reached."
output_iters_v[:, :, :, :] = output_iters_to_target

# ── Forcing-space metrics (L96 only) ─────────────────────────────────
if has_forcing
    forcing_samples_v = defVar(ds, "forcing_samples", Float64, ("random_seed", "ensemble_size", "k_iter", "pushforward_sample", "forcing_dim"); fillvalue=NaN)
    forcing_samples_v.attrib["description"] = "$(n_pushforward_samples) posterior pushforward samples in forcing space"
    forcing_samples_v[:, :, :, :, :] = forcing_samples_arr

    forcing_mahal_v = defVar(ds, "forcing_mahalanobis", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
    forcing_mahal_v.attrib["description"] = "Mahalanobis distance in forcing space: (fm-truth_forcing)' Cf^{-1} (fm-truth_forcing)."
    forcing_mahal_v[:, :, :] = forcing_mahal_arr

    forcing_logpdf_true_v_map_v = defVar(ds, "forcing_logpdf_true_v_map", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
    forcing_logpdf_true_v_map_v.attrib["description"] = "Log PDF ratio in forcing space: logpdf(N(fm,Cf), truth_forcing) - logpdf(N(fm,Cf), mode)."
    forcing_logpdf_true_v_map_v[:, :, :] = forcing_logpdf_true_v_map_arr

    forcing_plr_top_v = defVar(ds, "forcing_plr_mahalanobis_top", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
    forcing_plr_top_v.attrib["description"] = "PLR Mahalanobis top term in forcing space: sum((Vf'*(fm-truth))^2 / λf_i), top $(n_lowrank_modes) modes. Ref: Chisq($(n_lowrank_modes))."
    forcing_plr_top_v[:, :, :] = forcing_plr_mahal_top_arr

    forcing_plr_res_v = defVar(ds, "forcing_plr_mahalanobis_residual", Float64, ("random_seed", "ensemble_size", "k_iter"); fillvalue=NaN)
    forcing_plr_res_v.attrib["description"] = "PLR Mahalanobis residual term in forcing space: (||fm-truth||^2 - ||Vf'*(fm-truth)||^2) / a_f. Ref: Chisq($(n_forcing - n_lowrank_modes))."
    forcing_plr_res_v[:, :, :] = forcing_plr_mahal_residual_arr

    forcing_coverage_v = defVar(ds, "forcing_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue=NaN)
    forcing_coverage_v.attrib["description"] = "Marginal coverage in forcing space: fraction of forcing dims where truth_forcing[d] ≤ q_p of the $(n_pushforward_samples) pushforward samples."
    forcing_coverage_v[:, :, :, :] = forcing_coverage_arr

    forcing_budget_v = defVar(ds, "forcing_budget_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue=NaN)
    forcing_budget_v.attrib["description"] = "Budget (N_ens·k_iter) to first reach |S(q)−q| ≤ c·√(q(1−q)/N_y) per quantile q in forcing space. NaN = not reached. Take max over quantiles for the all-quantile condition."
    forcing_budget_v[:, :, :, :] = forcing_budget_to_target

    forcing_iters_v = defVar(ds, "forcing_iters_to_target", Float64, ("random_seed", "ensemble_size", "target_scaling", "coverage_quantile"); fillvalue=NaN)
    forcing_iters_v.attrib["description"] = "Iterations k_iter to first reach coverage target per quantile in forcing space. NaN = not reached."
    forcing_iters_v[:, :, :, :] = forcing_iters_to_target
end

close(ds)
@info "Saved leaderboard data to $(nc_save_filename)"
