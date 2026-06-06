using NCDatasets
using Distributions
using Statistics
using Printf
using DataFrames
using Dates

calib_date = Date("2026-06-03", "yyyy-mm-dd")
indir = joinpath("output","from-hpc_$(calib_date)")
#filename = "ces-eki-dmc_l63_ensemble_results_$(calib_date).nc"
#filename = "ces-eki-dmc_l96_ensemble_results_$(calib_date).nc"
filename = "ces-eki-dmc_l96_spatial_forcing_ensemble_results_$(calib_date).nc"
#filename = "ces-eki-dmc_l96_nn_forcing_ensemble_results_$(calib_date).nc"

filename = joinpath(indir, filename)
@info "computing leaderboard metrics from $(filename)"

###########################################################################
#################### Metric parameters ###################################
###########################################################################

calibration_check_quantiles = [0.1, 0.5, 0.9] # quantile levels for calibration check scores and coverage
n_lowrank_modes             = 5                # top EOF modes for perturbed-low-rank (aI+UDU') Mahalanobis

ncd = NCDataset(filename)
mh = ncd[:mahalanobis]
lp = ncd[:posterior_logpdf_true_v_map]
# Gaussian: mh = -2lp
(n_rng, n_ens_size, n_k_iter, n_par) = (ncd.dim[k] for k in ["random_seed","ensemble_size", "k_iter", "param_dim"])

mh_nonmiss = sum(.!ismissing.(mh), dims=1)[1,:,:]
lp_nonmiss = sum(.!ismissing.(lp), dims=1)[1,:,:]

qq_good  = quantile(Chisq(n_par), calibration_check_quantiles)

# === 1. Validate: pooled empirical quantiles should match chi-sq(n_par) ===
# If mh is correctly computed it follows chi-sq(n_par), so empirical ≈ theoretical.
mh_pool = collect(skipmissing(vec(Array(mh))))
lp_pool = collect(skipmissing(vec(-2 .* Array(lp))))

mh_emp_q = quantile(mh_pool, calibration_check_quantiles)
lp_emp_q = quantile(lp_pool, calibration_check_quantiles)

println("Metric calibration check — pooled empirical quantiles vs chi-sq($(n_par)) (should match if metric is correct):")
println(@sprintf("  %-8s  %-12s  %-12s  %-12s", "prob", "chi-sq ref", "mh", "-2*lp"))
for (i, p) in enumerate(calibration_check_quantiles)
    println(@sprintf("  %-8.2f  %-12.3f  %-12.3f  %-12.3f", p, qq_good[i], mh_emp_q[i], lp_emp_q[i]))
end

# === 2. Per-experiment scores ===
# missing where no valid seeds; * in printout where coverage < n_rng (less trusted)
mh_scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
lp_scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)

for (idx, qg) in enumerate(qq_good)
    for j in axes(mh, 3), i in axes(mh, 2)
        nm = mh_nonmiss[i, j]
        nm == 0 && continue
        mh_scores[idx, i, j] = sum(skipmissing(mh[:, i, j]) .<= qg; init=0) / nm
    end
    for j in axes(lp, 3), i in axes(lp, 2)
        nm = lp_nonmiss[i, j]
        nm == 0 && continue
        lp_scores[idx, i, j] = sum(skipmissing(-2 .* lp[:, i, j]) .<= qg; init=0) / nm
    end
end

mh_coverage = mh_nonmiss ./ n_rng   # 1.0 = all n_rng seeds valid; < 1 = partial
lp_coverage = lp_nonmiss ./ n_rng

# === 3. Store scores as DataFrame ===
ens_vals   = try Int.(ncd["ensemble_size"][:]) catch; collect(1:n_ens_size) end
kiter_vals = try Int.(ncd["k_iter"][:])        catch; collect(1:n_k_iter)   end

df_scores = DataFrame(
    q_prob      = Float64[],
    ens_size    = Int[],
    k_iter      = Int[],
    mh_score    = Union{Float64,Missing}[],
    lp_score    = Union{Float64,Missing}[],
    mh_coverage = Float64[],
    lp_coverage = Float64[],
)

for (qi, p) in enumerate(calibration_check_quantiles)
    for (ei, ev) in enumerate(ens_vals)
        for (ki, kv) in enumerate(kiter_vals)
            push!(df_scores, (
                q_prob      = p,
                ens_size    = ev,
                k_iter      = kv,
                mh_score    = mh_scores[qi, ei, ki],
                lp_score    = lp_scores[qi, ei, ki],
                mh_coverage = mh_coverage[ei, ki],
                lp_coverage = lp_coverage[ei, ki],
            ))
        end
    end
end

# rows with mh_coverage < 1 used fewer than n_rng seeds (less trusted)
println("\nScores by ens_size and q_prob (missing = no valid seeds; coverage < 1 = less trusted):")
println("  (Expected: score at each q ≈ q for a well-calibrated posterior)")
for ens in sort(unique(df_scores.ens_size))
    println("\n=== ens_size = $(ens) ===")
    for (qi, p) in enumerate(calibration_check_quantiles)
        sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_scores), :k_iter)
        println("  q = $(p)  [chi-sq($(n_par)) bound = $(round(qq_good[qi], digits=2)),  expected score ≈ $(p)]")
        show(stdout, select(sub, :k_iter, :mh_score, :lp_score, :mh_coverage), allrows=true)
        println()
    end
end

# === 4. Forcing-space M and L scores ===
if haskey(ncd, "forcing_mahalanobis") && haskey(ncd, "forcing_logpdf_true_v_map")
    n_for     = ncd.dim["forcing_dim"]
    fmh       = ncd[:forcing_mahalanobis]
    flp       = ncd[:forcing_logpdf_true_v_map]
    qq_good_f = quantile(Chisq(n_for), calibration_check_quantiles)

    fmh_nonmiss = sum(.!ismissing.(fmh), dims=1)[1,:,:]
    flp_nonmiss = sum(.!ismissing.(flp), dims=1)[1,:,:]
    fmh_pool    = collect(skipmissing(vec(Array(fmh))))
    flp_pool    = collect(skipmissing(vec(-2 .* Array(flp))))
    fmh_emp_q   = quantile(fmh_pool, calibration_check_quantiles)
    flp_emp_q   = quantile(flp_pool, calibration_check_quantiles)

    println("\n" * "="^72)
    println("Forcing-space M and L  (chi-sq ref dim = $(n_for))")
    println("="^72)
    println("Metric calibration check — pooled empirical quantiles vs chi-sq($(n_for)):")
    println(@sprintf("  %-8s  %-12s  %-12s  %-12s", "prob", "chi-sq ref", "mh_f", "-2*lp_f"))
    for (i, p) in enumerate(calibration_check_quantiles)
        println(@sprintf("  %-8.2f  %-12.3f  %-12.3f  %-12.3f", p, qq_good_f[i], fmh_emp_q[i], flp_emp_q[i]))
    end

    fmh_scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    flp_scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    for (idx, qg) in enumerate(qq_good_f)
        for j in axes(fmh, 3), i in axes(fmh, 2)
            nm = fmh_nonmiss[i, j]; nm == 0 && continue
            fmh_scores[idx, i, j] = sum(skipmissing(fmh[:, i, j]) .<= qg; init=0) / nm
        end
        for j in axes(flp, 3), i in axes(flp, 2)
            nm = flp_nonmiss[i, j]; nm == 0 && continue
            flp_scores[idx, i, j] = sum(skipmissing(-2 .* flp[:, i, j]) .<= qg; init=0) / nm
        end
    end

    fmh_coverage = fmh_nonmiss ./ n_rng
    flp_coverage = flp_nonmiss ./ n_rng
    df_fscores = DataFrame(
        q_prob=Float64[], ens_size=Int[], k_iter=Int[],
        mh_score=Union{Float64,Missing}[], lp_score=Union{Float64,Missing}[],
        mh_coverage=Float64[], lp_coverage=Float64[],
    )
    for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
        push!(df_fscores, (q_prob=p, ens_size=ev, k_iter=kv,
            mh_score=fmh_scores[qi,ei,ki], lp_score=flp_scores[qi,ei,ki],
            mh_coverage=fmh_coverage[ei,ki], lp_coverage=flp_coverage[ei,ki]))
    end

    println("\nForcing-space scores by ens_size and q_prob:")
    println("  (Expected: score at each q ≈ q for a well-calibrated posterior)")
    for ens in sort(unique(df_fscores.ens_size))
        println("\n=== ens_size = $(ens) ===")
        for (qi, p) in enumerate(calibration_check_quantiles)
            sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_fscores), :k_iter)
            println("  q = $(p)  [chi-sq($(n_for)) bound = $(round(qq_good_f[qi], digits=2)),  expected score ≈ $(p)]")
            show(stdout, select(sub, :k_iter, :mh_score, :lp_score, :mh_coverage), allrows=true)
            println()
        end
    end
end

# === 5. Output-space M and L scores ===
if haskey(ncd, "output_mahalanobis") && haskey(ncd, "output_logpdf_true_v_map")
    n_out     = ncd.dim["output_dim"]
    omh       = ncd[:output_mahalanobis]
    olp       = ncd[:output_logpdf_true_v_map]
    qq_good_o = quantile(Chisq(n_out), calibration_check_quantiles)

    omh_nonmiss = sum(.!ismissing.(omh), dims=1)[1,:,:]
    olp_nonmiss = sum(.!ismissing.(olp), dims=1)[1,:,:]
    omh_pool    = collect(skipmissing(vec(Array(omh))))
    olp_pool    = collect(skipmissing(vec(-2 .* Array(olp))))
    omh_emp_q   = quantile(omh_pool, calibration_check_quantiles)
    olp_emp_q   = quantile(olp_pool, calibration_check_quantiles)

    println("\n" * "="^72)
    println("Output-space M and L  (chi-sq ref dim = $(n_out))")
    println("="^72)
    println("Metric calibration check — pooled empirical quantiles vs chi-sq($(n_out)):")
    println(@sprintf("  %-8s  %-12s  %-12s  %-12s", "prob", "chi-sq ref", "mh_o", "-2*lp_o"))
    for (i, p) in enumerate(calibration_check_quantiles)
        println(@sprintf("  %-8.2f  %-12.3f  %-12.3f  %-12.3f", p, qq_good_o[i], omh_emp_q[i], olp_emp_q[i]))
    end

    omh_scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    olp_scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    for (idx, qg) in enumerate(qq_good_o)
        for j in axes(omh, 3), i in axes(omh, 2)
            nm = omh_nonmiss[i, j]; nm == 0 && continue
            omh_scores[idx, i, j] = sum(skipmissing(omh[:, i, j]) .<= qg; init=0) / nm
        end
        for j in axes(olp, 3), i in axes(olp, 2)
            nm = olp_nonmiss[i, j]; nm == 0 && continue
            olp_scores[idx, i, j] = sum(skipmissing(-2 .* olp[:, i, j]) .<= qg; init=0) / nm
        end
    end

    omh_coverage = omh_nonmiss ./ n_rng
    olp_coverage = olp_nonmiss ./ n_rng
    df_oscores = DataFrame(
        q_prob=Float64[], ens_size=Int[], k_iter=Int[],
        mh_score=Union{Float64,Missing}[], lp_score=Union{Float64,Missing}[],
        mh_coverage=Float64[], lp_coverage=Float64[],
    )
    for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
        push!(df_oscores, (q_prob=p, ens_size=ev, k_iter=kv,
            mh_score=omh_scores[qi,ei,ki], lp_score=olp_scores[qi,ei,ki],
            mh_coverage=omh_coverage[ei,ki], lp_coverage=olp_coverage[ei,ki]))
    end

    println("\nOutput-space scores by ens_size and q_prob:")
    println("  (Expected: score at each q ≈ q for a well-calibrated posterior)")
    for ens in sort(unique(df_oscores.ens_size))
        println("\n=== ens_size = $(ens) ===")
        for (qi, p) in enumerate(calibration_check_quantiles)
            sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_oscores), :k_iter)
            println("  q = $(p)  [chi-sq($(n_out)) bound = $(round(qq_good_o[qi], digits=2)),  expected score ≈ $(p)]")
            show(stdout, select(sub, :k_iter, :mh_score, :lp_score, :mh_coverage), allrows=true)
            println()
        end
    end
end

# === 6. Perturbed-low-rank Mahalanobis (C ≈ aI + U*D*U', k = n_lowrank_modes top modes) ===
# Each space yields three terms with separate reference distributions:
#   top      ~ Chisq(k)    — miscalibration in the k principal directions
#   residual ~ Chisq(n-k)  — miscalibration in the n-k noise-floor directions (a = mean of bottom n-k eigenvalues)
#   total    ~ Chisq(n)    — full-space calibration (top + residual, identical to Woodbury inverse applied to full diff)
for (space_label, top_sym, res_sym, n_dim_key) in [
    ("Forcing", :forcing_plr_mahalanobis_top, :forcing_plr_mahalanobis_residual, "forcing_dim"),
    ("Output",  :output_plr_mahalanobis_top,  :output_plr_mahalanobis_residual,  "output_dim"),
]
    (haskey(ncd, top_sym) && haskey(ncd, res_sym)) || continue
    n_dim   = ncd.dim[n_dim_key]
    top_raw = Array(ncd[top_sym])   # (n_rng, n_ens_size, n_k_iter), Union{Float64,Missing}
    res_raw = Array(ncd[res_sym])
    tot_raw = top_raw .+ res_raw    # missing propagates where either component is missing

    println("\n" * "="^72)
    println("$(space_label)-space perturbed-low-rank M  (k=$(n_lowrank_modes) modes, n=$(n_dim) total dims)")
    println("  Covariance model: C ≈ a*I + U*D*U'  (a = mean of bottom $(n_dim-n_lowrank_modes) eigenvalues of empirical C)")
    println("  top      ~ Chisq($(n_lowrank_modes))        — miscalibration in the k principal directions")
    println("  residual ~ Chisq($(n_dim-n_lowrank_modes))  — miscalibration in the noise-floor directions")
    println("  total    ~ Chisq($(n_dim))                  — full-space calibration (top + residual)")
    println("="^72)

    for (part_label, arr, ref_dof) in [
        ("top      (k=$(n_lowrank_modes))",         top_raw, n_lowrank_modes),
        ("residual (n-k=$(n_dim-n_lowrank_modes))", res_raw, n_dim - n_lowrank_modes),
        ("total    (n=$(n_dim))",                   tot_raw, n_dim),
    ]
        nonmiss = sum(.!ismissing.(arr), dims=1)[1,:,:]
        pool    = collect(skipmissing(vec(arr)))
        isempty(pool) && continue
        qq_ref  = quantile(Chisq(ref_dof), calibration_check_quantiles)
        emp_q   = quantile(pool, calibration_check_quantiles)
        coverage = nonmiss ./ n_rng

        println("\n  $(space_label)-space PLR-M $(part_label)  [ref: Chisq($(ref_dof))]:")
        println("  Calibration check — pooled empirical quantiles vs chi-sq($(ref_dof)):")
        println(@sprintf("  %-8s  %-12s  %-12s", "prob", "chi-sq ref", "plr_mh"))
        for (i, p) in enumerate(calibration_check_quantiles)
            println(@sprintf("  %-8.2f  %-12.3f  %-12.3f", p, qq_ref[i], emp_q[i]))
        end

        scores = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
        for (idx, qg) in enumerate(qq_ref)
            for j in axes(arr, 3), i in axes(arr, 2)
                nm = nonmiss[i, j]; nm == 0 && continue
                scores[idx, i, j] = sum(skipmissing(arr[:, i, j]) .<= qg; init=0) / nm
            end
        end

        df_plr = DataFrame(
            q_prob   = Float64[],
            ens_size = Int[],
            k_iter   = Int[],
            score    = Union{Float64,Missing}[],
            coverage = Float64[],
        )
        for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
            push!(df_plr, (q_prob=p, ens_size=ev, k_iter=kv,
                score=scores[qi,ei,ki], coverage=coverage[ei,ki]))
        end

        println("\n  Scores by ens_size and q_prob  (expected score ≈ q_prob):")
        for ens in sort(unique(df_plr.ens_size))
            println("\n  === ens_size = $(ens) ===")
            for (qi, p) in enumerate(calibration_check_quantiles)
                sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_plr), :k_iter)
                println("    q = $(p)  [chi-sq($(ref_dof)) bound = $(round(qq_ref[qi], digits=2)),  expected score ≈ $(p)]")
                show(stdout, select(sub, :k_iter, :score, :coverage), allrows=true)
                println()
            end
        end
    end
end

# === 7. Marginal coverage fractions (forcing and output) ===
for (space_label, cov_sym) in [("Forcing", :forcing_coverage), ("Output", :output_coverage)]
    haskey(ncd, cov_sym) || continue
    cov    = ncd[cov_sym]          # (n_rng, n_ens_size, n_k_iter, n_cov_q)
    cov_qs = ncd["coverage_quantile"][:]

    println("\n" * "="^72)
    println("$(space_label)-space marginal coverage fractions")
    println("  (Spatial dims as trials; spatially correlated → effective n < stated n)")
    println("="^72)
    println("Pooled mean coverage across all experiments:")
    for (qi, qp) in enumerate(cov_qs)
        pool = collect(skipmissing(vec(Array(cov[:, :, :, qi]))))
        println(@sprintf("  q = %.1f  mean = %.3f  (expected %.3f)", qp, mean(pool), qp))
    end

    df_cov = DataFrame(
        q_prob        = Float64[],
        ens_size      = Int[],
        k_iter        = Int[],
        mean_coverage = Union{Float64,Missing}[],
        n_valid       = Int[],
    )
    for (qi, qp) in enumerate(cov_qs), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
        vals = collect(skipmissing(cov[:, ei, ki, qi]))
        push!(df_cov, (q_prob=qp, ens_size=ev, k_iter=kv,
            mean_coverage = isempty(vals) ? missing : mean(vals),
            n_valid       = length(vals)))
    end

    println("\n$(space_label)-space coverage by ens_size and q_prob:")
    println("  (Expected: mean_coverage ≈ q_prob for calibrated marginals)")
    for ens in sort(unique(df_cov.ens_size))
        println("\n=== ens_size = $(ens) ===")
        for (qi, qp) in enumerate(cov_qs)
            sub = sort(filter(r -> r.ens_size == ens && r.q_prob == qp, df_cov), :k_iter)
            println("  q = $(qp)  (expected ≈ $(qp))")
            show(stdout, select(sub, :k_iter, :mean_coverage, :n_valid), allrows=true)
            println()
        end
    end
end
