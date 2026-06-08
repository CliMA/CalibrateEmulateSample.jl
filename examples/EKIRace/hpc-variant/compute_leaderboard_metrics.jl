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

calibration_check_quantiles = [0.2, 0.5, 0.8] # quantile levels for calibration check scores and coverage
n_lowrank_modes             = 5                # top EOF modes for perturbed-low-rank (aI+UDU') Mahalanobis

###########################################################################
#################### Display filters ####################################
###########################################################################
#
#  TWO independent filter axes — set either to `nothing` to show everything
#  available in the file, or to a subset vector to restrict output.
#
#  display_spaces — which spaces to evaluate metrics in:
#    :param    — parameter space   (posterior mean vs true parameter)
#    :forcing  — forcing space     (pushforward posterior vs true forcing)
#    :output   — output space      (pushforward posterior vs observations)
#
#  display_metrics — which metric types to display:
#    :mahalanobis     — full Mahalanobis distance  M = (m−truth)ᵀ C⁻¹ (m−truth)
#    :logratio        — log PDF ratio              L = log p(truth|post) − log p(mode|post)
#    :plr_mahalanobis — perturbed-low-rank M        (top / residual / total; forcing and output only)
#    :coverage        — marginal coverage fractions (forcing and output only)
#
#  display_ens_sizes — which ensemble-size values (integers) to show in tables.

display_spaces    = [:output]   # e.g. [:param, :forcing]
display_metrics   = [:coverage]  # e.g. [:mahalanobis, :plr_mahalanobis]
display_ens_sizes = nothing   # e.g. [10, 50]

show_metric(m) = isnothing(display_metrics)  || m in display_metrics
show_space(s)  = isnothing(display_spaces)   || s in display_spaces
show_ens(e)    = isnothing(display_ens_sizes) || e in display_ens_sizes

###########################################################################
#################### Load file and report available options ##############
###########################################################################

ncd = NCDataset(filename)
mh = ncd[:mahalanobis]
lp = ncd[:posterior_logpdf_true_v_map]
(n_rng, n_ens_size, n_k_iter, n_par) = (ncd.dim[k] for k in ["random_seed","ensemble_size", "k_iter", "param_dim"])
ens_vals   = try Int.(ncd["ensemble_size"][:]) catch; collect(1:n_ens_size) end
kiter_vals = try Int.(ncd["k_iter"][:])        catch; collect(1:n_k_iter)   end

# Discover what is actually present in this file
avail_spaces  = Symbol[:param]
avail_metrics = Symbol[:mahalanobis, :logratio]
(haskey(ncd, "forcing_mahalanobis") || haskey(ncd, "forcing_plr_mahalanobis_top") || haskey(ncd, "forcing_coverage")) && push!(avail_spaces, :forcing)
(haskey(ncd, "output_mahalanobis")  || haskey(ncd, "output_plr_mahalanobis_top")  || haskey(ncd, "output_coverage"))  && push!(avail_spaces, :output)
(haskey(ncd, "forcing_plr_mahalanobis_top") || haskey(ncd, "output_plr_mahalanobis_top")) && push!(avail_metrics, :plr_mahalanobis)
(haskey(ncd, "forcing_coverage") || haskey(ncd, "output_coverage"))                        && push!(avail_metrics, :coverage)

println("="^72)
println("File: $(filename)")
println()
println("All available options:")
println("  display_spaces    — $(avail_spaces)")
println("  display_metrics   — $(avail_metrics)")
println("  display_ens_sizes — $(ens_vals)")
println()
println("Currently selected (edit display_* variables above to filter):")
println("  display_spaces    = $(isnothing(display_spaces)    ? "nothing  →  $(avail_spaces)"  : display_spaces)")
println("  display_metrics   = $(isnothing(display_metrics)   ? "nothing  →  $(avail_metrics)" : display_metrics)")
println("  display_ens_sizes = $(isnothing(display_ens_sizes) ? "nothing  →  $(ens_vals)"       : display_ens_sizes)")
println("="^72)

mh_nonmiss  = sum(.!ismissing.(mh), dims=1)[1,:,:]
lp_nonmiss  = sum(.!ismissing.(lp), dims=1)[1,:,:]
mh_coverage = mh_nonmiss ./ n_rng
lp_coverage = lp_nonmiss ./ n_rng
qq_good     = quantile(Chisq(n_par), calibration_check_quantiles)

###########################################################################
#################### Param-space: Mahalanobis and log-ratio ##############
###########################################################################

if show_space(:param) && (show_metric(:mahalanobis) || show_metric(:logratio))

    println("\n" * "="^72)
    println("Param-space  (chi-sq ref dim = $(n_par))")
    println("="^72)

    mh_pool  = collect(skipmissing(vec(Array(mh))))
    lp_pool  = collect(skipmissing(vec(-2 .* Array(lp))))
    mh_emp_q = quantile(mh_pool, calibration_check_quantiles)
    lp_emp_q = quantile(lp_pool, calibration_check_quantiles)

    println("Pooled-empirical calibration check vs chi-sq($(n_par))  (should match if metric is correct):")
    hdr = @sprintf("  %-8s  %-12s", "prob", "chi-sq ref")
    show_metric(:mahalanobis) && (hdr *= @sprintf("  %-12s", "mh"))
    show_metric(:logratio)    && (hdr *= @sprintf("  %-12s", "-2*lp"))
    println(hdr)
    for (i, p) in enumerate(calibration_check_quantiles)
        row = @sprintf("  %-8.2f  %-12.3f", p, qq_good[i])
        show_metric(:mahalanobis) && (row *= @sprintf("  %-12.3f", mh_emp_q[i]))
        show_metric(:logratio)    && (row *= @sprintf("  %-12.3f", lp_emp_q[i]))
        println(row)
    end

    mh_scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    lp_scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    mh_scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    lp_scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    for (idx, qg) in enumerate(qq_good)
        for j in axes(mh, 3), i in axes(mh, 2)
            nm = mh_nonmiss[i, j]; nm == 0 && continue
            ind = collect(skipmissing(mh[:, i, j])) .<= qg
            mh_scores[idx, i, j]     = mean(ind)
            mh_scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
        end
        for j in axes(lp, 3), i in axes(lp, 2)
            nm = lp_nonmiss[i, j]; nm == 0 && continue
            ind = collect(skipmissing(-2 .* lp[:, i, j])) .<= qg
            lp_scores[idx, i, j]     = mean(ind)
            lp_scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
        end
    end

    df_param = DataFrame(
        q_prob       = Float64[],
        ens_size     = Int[],
        k_iter       = Int[],
        mh_score     = Union{Float64,Missing}[],
        mh_score_std = Union{Float64,Missing}[],
        lp_score     = Union{Float64,Missing}[],
        lp_score_std = Union{Float64,Missing}[],
        mh_coverage  = Float64[],
        lp_coverage  = Float64[],
    )
    for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
        push!(df_param, (
            q_prob       = p, ens_size     = ev, k_iter       = kv,
            mh_score     = mh_scores[qi, ei, ki],
            mh_score_std = mh_scores_std[qi, ei, ki],
            lp_score     = lp_scores[qi, ei, ki],
            lp_score_std = lp_scores_std[qi, ei, ki],
            mh_coverage  = mh_coverage[ei, ki],
            lp_coverage  = lp_coverage[ei, ki],
        ))
    end

    score_cols = Symbol[:k_iter]
    show_metric(:mahalanobis) && append!(score_cols, [:mh_score, :mh_score_std, :mh_coverage])
    show_metric(:logratio)    && append!(score_cols, [:lp_score, :lp_score_std, :lp_coverage])

    println("\nScores by ens_size and k_iter  (missing = no valid seeds; _coverage < 1 = less trusted)")
    println("  Expected: score at each q ≈ q for a well-calibrated posterior")
    for ens in filter(show_ens, sort(unique(df_param.ens_size)))
        println("\n=== ens_size = $(ens) ===")
        for (qi, p) in enumerate(calibration_check_quantiles)
            sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_param), :k_iter)
            println("  q = $(p)  [chi-sq($(n_par)) bound = $(round(qq_good[qi], digits=2)),  expected score ≈ $(p)]")
            show(stdout, select(sub, score_cols), allrows=true)
            println()
        end
    end
end

###########################################################################
#################### Forcing-space: Mahalanobis and log-ratio ############
###########################################################################

if show_space(:forcing) && (show_metric(:mahalanobis) || show_metric(:logratio)) &&
        haskey(ncd, "forcing_mahalanobis") && haskey(ncd, "forcing_logpdf_true_v_map")

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
    println("Forcing-space  (chi-sq ref dim = $(n_for))")
    println("="^72)
    println("Pooled-empirical calibration check vs chi-sq($(n_for)):")
    hdr = @sprintf("  %-8s  %-12s", "prob", "chi-sq ref")
    show_metric(:mahalanobis) && (hdr *= @sprintf("  %-12s", "mh_f"))
    show_metric(:logratio)    && (hdr *= @sprintf("  %-12s", "-2*lp_f"))
    println(hdr)
    for (i, p) in enumerate(calibration_check_quantiles)
        row = @sprintf("  %-8.2f  %-12.3f", p, qq_good_f[i])
        show_metric(:mahalanobis) && (row *= @sprintf("  %-12.3f", fmh_emp_q[i]))
        show_metric(:logratio)    && (row *= @sprintf("  %-12.3f", flp_emp_q[i]))
        println(row)
    end

    fmh_scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    flp_scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    fmh_scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    flp_scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    for (idx, qg) in enumerate(qq_good_f)
        for j in axes(fmh, 3), i in axes(fmh, 2)
            nm = fmh_nonmiss[i, j]; nm == 0 && continue
            ind = collect(skipmissing(fmh[:, i, j])) .<= qg
            fmh_scores[idx, i, j]     = mean(ind)
            fmh_scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
        end
        for j in axes(flp, 3), i in axes(flp, 2)
            nm = flp_nonmiss[i, j]; nm == 0 && continue
            ind = collect(skipmissing(-2 .* flp[:, i, j])) .<= qg
            flp_scores[idx, i, j]     = mean(ind)
            flp_scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
        end
    end

    fmh_coverage = fmh_nonmiss ./ n_rng
    flp_coverage = flp_nonmiss ./ n_rng
    df_fscores = DataFrame(
        q_prob=Float64[], ens_size=Int[], k_iter=Int[],
        mh_score=Union{Float64,Missing}[], mh_score_std=Union{Float64,Missing}[],
        lp_score=Union{Float64,Missing}[], lp_score_std=Union{Float64,Missing}[],
        mh_coverage=Float64[], lp_coverage=Float64[],
    )
    for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
        push!(df_fscores, (q_prob=p, ens_size=ev, k_iter=kv,
            mh_score=fmh_scores[qi,ei,ki], mh_score_std=fmh_scores_std[qi,ei,ki],
            lp_score=flp_scores[qi,ei,ki], lp_score_std=flp_scores_std[qi,ei,ki],
            mh_coverage=fmh_coverage[ei,ki], lp_coverage=flp_coverage[ei,ki]))
    end

    score_cols = Symbol[:k_iter]
    show_metric(:mahalanobis) && append!(score_cols, [:mh_score, :mh_score_std, :mh_coverage])
    show_metric(:logratio)    && append!(score_cols, [:lp_score, :lp_score_std, :lp_coverage])

    println("\nScores by ens_size and k_iter  (expected score ≈ q_prob for calibrated posterior)")
    for ens in filter(show_ens, sort(unique(df_fscores.ens_size)))
        println("\n=== ens_size = $(ens) ===")
        for (qi, p) in enumerate(calibration_check_quantiles)
            sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_fscores), :k_iter)
            println("  q = $(p)  [chi-sq($(n_for)) bound = $(round(qq_good_f[qi], digits=2)),  expected score ≈ $(p)]")
            show(stdout, select(sub, score_cols), allrows=true)
            println()
        end
    end
end

###########################################################################
#################### Output-space: Mahalanobis and log-ratio #############
###########################################################################

if show_space(:output) && (show_metric(:mahalanobis) || show_metric(:logratio)) &&
        haskey(ncd, "output_mahalanobis") && haskey(ncd, "output_logpdf_true_v_map")

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
    println("Output-space  (chi-sq ref dim = $(n_out))")
    println("="^72)
    println("Pooled-empirical calibration check vs chi-sq($(n_out)):")
    hdr = @sprintf("  %-8s  %-12s", "prob", "chi-sq ref")
    show_metric(:mahalanobis) && (hdr *= @sprintf("  %-12s", "mh_o"))
    show_metric(:logratio)    && (hdr *= @sprintf("  %-12s", "-2*lp_o"))
    println(hdr)
    for (i, p) in enumerate(calibration_check_quantiles)
        row = @sprintf("  %-8.2f  %-12.3f", p, qq_good_o[i])
        show_metric(:mahalanobis) && (row *= @sprintf("  %-12.3f", omh_emp_q[i]))
        show_metric(:logratio)    && (row *= @sprintf("  %-12.3f", olp_emp_q[i]))
        println(row)
    end

    omh_scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    olp_scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    omh_scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    olp_scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
    for (idx, qg) in enumerate(qq_good_o)
        for j in axes(omh, 3), i in axes(omh, 2)
            nm = omh_nonmiss[i, j]; nm == 0 && continue
            ind = collect(skipmissing(omh[:, i, j])) .<= qg
            omh_scores[idx, i, j]     = mean(ind)
            omh_scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
        end
        for j in axes(olp, 3), i in axes(olp, 2)
            nm = olp_nonmiss[i, j]; nm == 0 && continue
            ind = collect(skipmissing(-2 .* olp[:, i, j])) .<= qg
            olp_scores[idx, i, j]     = mean(ind)
            olp_scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
        end
    end

    omh_coverage = omh_nonmiss ./ n_rng
    olp_coverage = olp_nonmiss ./ n_rng
    df_oscores = DataFrame(
        q_prob=Float64[], ens_size=Int[], k_iter=Int[],
        mh_score=Union{Float64,Missing}[], mh_score_std=Union{Float64,Missing}[],
        lp_score=Union{Float64,Missing}[], lp_score_std=Union{Float64,Missing}[],
        mh_coverage=Float64[], lp_coverage=Float64[],
    )
    for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
        push!(df_oscores, (q_prob=p, ens_size=ev, k_iter=kv,
            mh_score=omh_scores[qi,ei,ki], mh_score_std=omh_scores_std[qi,ei,ki],
            lp_score=olp_scores[qi,ei,ki], lp_score_std=olp_scores_std[qi,ei,ki],
            mh_coverage=omh_coverage[ei,ki], lp_coverage=olp_coverage[ei,ki]))
    end

    score_cols = Symbol[:k_iter]
    show_metric(:mahalanobis) && append!(score_cols, [:mh_score, :mh_score_std, :mh_coverage])
    show_metric(:logratio)    && append!(score_cols, [:lp_score, :lp_score_std, :lp_coverage])

    println("\nScores by ens_size and k_iter  (expected score ≈ q_prob for calibrated posterior)")
    for ens in filter(show_ens, sort(unique(df_oscores.ens_size)))
        println("\n=== ens_size = $(ens) ===")
        for (qi, p) in enumerate(calibration_check_quantiles)
            sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_oscores), :k_iter)
            println("  q = $(p)  [chi-sq($(n_out)) bound = $(round(qq_good_o[qi], digits=2)),  expected score ≈ $(p)]")
            show(stdout, select(sub, score_cols), allrows=true)
            println()
        end
    end
end

###########################################################################
#################### Perturbed-low-rank Mahalanobis ######################
###########################################################################
# C ≈ aI + U*D*U' (k = n_lowrank_modes top modes). Three terms per space:
#   top      ~ Chisq(k)    — miscalibration in the k principal directions
#   residual ~ Chisq(n-k)  — miscalibration in the noise-floor directions
#   total    ~ Chisq(n)    — full-space (= top + residual)

if show_metric(:plr_mahalanobis)
    for (space_label, space_sym, top_sym, res_sym, n_dim_key) in [
        ("Forcing", :forcing, :forcing_plr_mahalanobis_top, :forcing_plr_mahalanobis_residual, "forcing_dim"),
        ("Output",  :output,  :output_plr_mahalanobis_top,  :output_plr_mahalanobis_residual,  "output_dim"),
    ]
        (show_space(space_sym) && haskey(ncd, top_sym) && haskey(ncd, res_sym)) || continue

        n_dim   = ncd.dim[n_dim_key]
        top_raw = Array(ncd[top_sym])
        res_raw = Array(ncd[res_sym])
        tot_raw = top_raw .+ res_raw    # missing propagates where either component is missing

        println("\n" * "="^72)
        println("$(space_label)-space perturbed-low-rank Mahalanobis  (k=$(n_lowrank_modes) modes, n=$(n_dim) total dims)")
        println("  Covariance model: C ≈ a·I + U·D·Uᵀ  (a = mean of bottom $(n_dim-n_lowrank_modes) eigenvalues of C)")
        println("  top      ~ Chisq($(n_lowrank_modes))        — miscalibration in the k principal directions")
        println("  residual ~ Chisq($(n_dim-n_lowrank_modes))  — miscalibration in the noise-floor directions")
        println("  total    ~ Chisq($(n_dim))                  — full-space (top + residual)")
        println("="^72)

        for (part_label, arr, ref_dof) in [
            ("top      (k=$(n_lowrank_modes))",          top_raw, n_lowrank_modes),
            ("residual (n-k=$(n_dim-n_lowrank_modes))",  res_raw, n_dim - n_lowrank_modes),
            ("total    (n=$(n_dim))",                    tot_raw, n_dim),
        ]
            nonmiss  = sum(.!ismissing.(arr), dims=1)[1,:,:]
            pool     = collect(skipmissing(vec(arr)))
            isempty(pool) && continue
            qq_ref   = quantile(Chisq(ref_dof), calibration_check_quantiles)
            emp_q    = quantile(pool, calibration_check_quantiles)
            coverage = nonmiss ./ n_rng

            println("\n  $(space_label)-space PLR-M $(part_label)  [ref: Chisq($(ref_dof))]:")
            println("  Pooled-empirical calibration check vs chi-sq($(ref_dof)):")
            println(@sprintf("  %-8s  %-12s  %-12s", "prob", "chi-sq ref", "plr_mh"))
            for (i, p) in enumerate(calibration_check_quantiles)
                println(@sprintf("  %-8.2f  %-12.3f  %-12.3f", p, qq_ref[i], emp_q[i]))
            end

            scores     = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
            scores_std = Array{Union{Float64,Missing}}(missing, length(calibration_check_quantiles), n_ens_size, n_k_iter)
            for (idx, qg) in enumerate(qq_ref)
                for j in axes(arr, 3), i in axes(arr, 2)
                    nm = nonmiss[i, j]; nm == 0 && continue
                    ind = collect(skipmissing(arr[:, i, j])) .<= qg
                    scores[idx, i, j]     = mean(ind)
                    scores_std[idx, i, j] = nm > 1 ? std(ind) : missing
                end
            end

            df_plr = DataFrame(
                q_prob    = Float64[],
                ens_size  = Int[],
                k_iter    = Int[],
                score     = Union{Float64,Missing}[],
                score_std = Union{Float64,Missing}[],
                coverage  = Float64[],
            )
            for (qi, p) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
                push!(df_plr, (q_prob=p, ens_size=ev, k_iter=kv,
                    score=scores[qi,ei,ki], score_std=scores_std[qi,ei,ki], coverage=coverage[ei,ki]))
            end

            println("\n  Scores by ens_size and k_iter  (expected score ≈ q_prob):")
            for ens in filter(show_ens, sort(unique(df_plr.ens_size)))
                println("\n  === ens_size = $(ens) ===")
                for (qi, p) in enumerate(calibration_check_quantiles)
                    sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_plr), :k_iter)
                    println("    q = $(p)  [chi-sq($(ref_dof)) bound = $(round(qq_ref[qi], digits=2)),  expected score ≈ $(p)]")
                    show(stdout, select(sub, :k_iter, :score, :score_std, :coverage), allrows=true)
                    println()
                end
            end
        end
    end
end

###########################################################################
#################### Marginal coverage fractions #########################
###########################################################################

if show_metric(:coverage)
    for (space_label, space_sym, cov_sym) in [
        ("Forcing", :forcing, :forcing_coverage),
        ("Output",  :output,  :output_coverage),
    ]
        (show_space(space_sym) && haskey(ncd, cov_sym)) || continue

        cov        = ncd[cov_sym]          # (n_rng, n_ens_size, n_k_iter, n_cov_q)
        cov_qs_all = Float64.(ncd["coverage_quantile"][:])
        qi_keep    = findall(qp -> any(isapprox(qp, p, atol=1e-8) for p in calibration_check_quantiles), cov_qs_all)
        cov_qs     = cov_qs_all[qi_keep]

        println("\n" * "="^72)
        println("$(space_label)-space marginal coverage fractions")
        println("  (Spatial dims as trials; spatially correlated → effective n < stated n)")
        println("="^72)
        println("Pooled mean coverage across all experiments:")
        for (qi_local, qi_file) in enumerate(qi_keep)
            qp   = cov_qs[qi_local]
            pool = collect(skipmissing(vec(Array(cov[:, :, :, qi_file]))))
            println(@sprintf("  q = %.2f  mean = %.3f  (expected %.3f)", qp, mean(pool), qp))
        end

        df_cov = DataFrame(
            q_prob        = Float64[],
            ens_size      = Int[],
            k_iter        = Int[],
            mean_coverage = Union{Float64,Missing}[],
            std_coverage  = Union{Float64,Missing}[],
            n_valid       = Int[],
        )
        for (qi_local, qi_file) in enumerate(qi_keep), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
            qp   = cov_qs[qi_local]
            vals = collect(skipmissing(cov[:, ei, ki, qi_file]))
            push!(df_cov, (q_prob=qp, ens_size=ev, k_iter=kv,
                mean_coverage = isempty(vals) ? missing : mean(vals),
                std_coverage  = length(vals) > 1 ? std(vals) : missing,
                n_valid       = length(vals)))
        end

        println("\n$(space_label)-space coverage by ens_size and k_iter:")
        println("  (Expected: mean_coverage ≈ q_prob for calibrated marginals)")
        for ens in filter(show_ens, sort(unique(df_cov.ens_size)))
            println("\n=== ens_size = $(ens) ===")
            for (qi, qp) in enumerate(cov_qs)
                sub = sort(filter(r -> r.ens_size == ens && r.q_prob == qp, df_cov), :k_iter)
                println("  q = $(qp)  (expected ≈ $(qp))")
                show(stdout, select(sub, :k_iter, :mean_coverage, :std_coverage, :n_valid), allrows=true)
                println()
            end
        end
    end
end

@info calibration_check_quantiles
