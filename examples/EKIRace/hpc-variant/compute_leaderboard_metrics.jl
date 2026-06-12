using NCDatasets
using Distributions
using Statistics
using Printf
using DataFrames
using Dates
using JLD2
using LinearAlgebra

calib_date = Date("2026-06-11", "yyyy-mm-dd")
indir = joinpath("output","from-hpc_$(calib_date)")

experiment_list = [:l63, :l96_const, :l96_vec, :l96_flux ]
experiment = experiment_list[1] 
if experiment == :l63
    filename = "ces-eki-dmc_l63_ensemble_results_$(calib_date).nc"
    prelim_jld2_file = "l63_computed_preliminaries.jld2"
elseif experiment == :l96_const
    filename = "ces-eki-dmc_l96_ensemble_results_$(calib_date).nc"
    prelim_jld2_file = "l96_computed_preliminaries_const-force.jld2"
elseif experiment == :l96_vec
    filename = "ces-eki-dmc_l96_spatial_forcing_ensemble_results_$(calib_date).nc"
    prelim_jld2_file = "l96_computed_preliminaries_vec-force.jld2"
elseif experiment == :l96_flux
    filename = "ces-eki-dmc_l96_nn_forcing_ensemble_results_$(calib_date).nc"
    prelim_jld2_file = "l96_computed_preliminaries_flux-force.jld2"
else
    throw(ArgumentError("Expected Experiment from $(experiment_list). Got $(experiment)."))
end

filename = joinpath(indir, filename)
prelim_filename = joinpath(indir, prelim_jld2_file)
# Optional: path to the prelim JLD2 written by calibrate_l96.jl.
# Enables coverage recomputation at any quantile and R-whitened PCA coverage.
# e.g. "output/l96_computed_preliminaries_const-force.jld2"



###########################################################################
#################### Metric parameters ###################################
###########################################################################

calibration_check_quantiles = [0.15, 0.5, 0.85] # quantile levels for calibration check scores and coverage
n_lowrank_modes             = 2              # top EOF modes for perturbed-low-rank (aI+UDU') Mahalanobis
R_variance_retain           = 0.99             # fraction of R variance to retain for R-whitened PCA coverage


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
display_metrics   = [:coverage, :mahalanobis]  # e.g. [:mahalanobis, :plr_mahalanobis]
display_ens_sizes = nothing   # e.g. [10, 50]

show_metric(m) = isnothing(display_metrics)  || m in display_metrics
show_space(s)  = isnothing(display_spaces)   || s in display_spaces
show_ens(e)    = isnothing(display_ens_sizes) || e in display_ens_sizes

###########################################################################
#################### Load file and report available options ##############
###########################################################################

@info "computing leaderboard metrics from $(filename)"
ncd = NCDataset(filename)
mh = ncd[:mahalanobis]
lp = ncd[:posterior_logpdf_true_v_map]
(n_rng, n_ens_size, n_k_iter, n_par) = (ncd.dim[k] for k in ["random_seed","ensemble_size", "k_iter", "param_dim"])
ens_vals   = try Int.(ncd["ensemble_size"][:]) catch; collect(1:n_ens_size) end
kiter_vals = try Int.(ncd["k_iter"][:])        catch; collect(1:n_k_iter)   end

# Load truth and observation-noise covariance from the prelim JLD2 (calibrate_l96.jl output).
# Falls back to truth_output stored in the netcdf (legacy) if JLD2 is not configured.
if !isnothing(prelim_jld2_file) && isfile(prelim_filename)
    @info "using precomputed quantities from $(prelim_filename)"
    _pd     = JLD2.load(prelim_filename)
    y_truth = Float64.(_pd["y"])
    R_obs   = Float64.(_pd["R"])
    @info "Loaded y and R from $(prelim_jld2_file)"
elseif haskey(ncd, "truth_output")
    y_truth = Float64.(ncd["truth_output"][:])
    R_obs   = nothing
else
    @error "precomputed quantities NOT FOUND at $(prelim_filename)"
    y_truth = nothing
    R_obs   = nothing
end

# Discover what is actually present in this file
avail_spaces  = Symbol[:param]
avail_metrics = Symbol[:mahalanobis, :logratio]
(haskey(ncd, "forcing_mahalanobis") || haskey(ncd, "forcing_plr_mahalanobis_top") || haskey(ncd, "forcing_coverage") || (haskey(ncd, "forcing_samples") && haskey(ncd, "truth_forcing"))) && push!(avail_spaces, :forcing)
(haskey(ncd, "output_mahalanobis")  || haskey(ncd, "output_plr_mahalanobis_top")  || haskey(ncd, "output_coverage")  || (haskey(ncd, "output_samples") && !isnothing(y_truth)))          && push!(avail_spaces, :output)
(haskey(ncd, "forcing_plr_mahalanobis_top") || haskey(ncd, "output_plr_mahalanobis_top")) && push!(avail_metrics, :plr_mahalanobis)
((haskey(ncd, "forcing_samples") && haskey(ncd, "truth_forcing")) || (haskey(ncd, "output_samples") && !isnothing(y_truth)) || haskey(ncd, "forcing_coverage") || haskey(ncd, "output_coverage")) && push!(avail_metrics, :coverage)

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
# Coverage is computed on-the-fly from raw pushforward samples and truth,
# allowing any quantile level.  When R_obs is loaded from the prelim JLD2,
# an additional R-whitened (PCA) coverage is computed: each coordinate is
# decorrelated by the eigenvectors of R and scaled by the inverse sqrt of
# the corresponding eigenvalue, making dimensions more independent.

if show_metric(:coverage)

    # ── Output-space coverage ──────────────────────────────────────────────
    @info show_space(:output)
    @info haskey(ncd, "output_samples")
    @info !isnothing(y_truth)
    if show_space(:output) && haskey(ncd, "output_samples") && !isnothing(y_truth)

        n_out  = ncd.dim["output_dim"]
        os_all = coalesce.(Array(ncd["output_samples"]), NaN)   # (n_rng, n_ens_size, n_k_iter, n_ps, n_out)
        actual_out_dim = size(os_all, 5)
        actual_out_dim != n_out && @warn "output_samples trailing dim ($actual_out_dim) ≠ output_dim ($n_out); raw coverage uses actual dim, R-whitened coverage disabled"

        do_whiten = !isnothing(R_obs)
        if do_whiten
            eig_R   = eigen(Symmetric(R_obs))
            ord     = sortperm(eig_R.values; rev=true)   # descending eigenvalue order
            λ_all   = eig_R.values[ord]
            V_all   = eig_R.vectors[:, ord]
            cum_var = cumsum(λ_all) ./ sum(λ_all)
            k_R     = something(findfirst(>=(R_variance_retain), cum_var), length(λ_all))
            V_R     = V_all[:, 1:k_R]
            λ_R     = λ_all[1:k_R]
            yw      = V_R' * y_truth ./ sqrt.(λ_R)
            println(@sprintf("\nR eigenvalue truncation: %d / %d modes retained  (threshold %.1f%% → %.4f%% variance)",
                             k_R, n_out, 100*R_variance_retain, 100*cum_var[k_R]))
        end

        for (label, do_white) in [("raw", false), ("R-whitened PCA", true)]
            (do_white && !do_whiten) && continue

            local cov_arr = fill(NaN, n_rng, n_ens_size, n_k_iter, length(calibration_check_quantiles))
            for ri in 1:n_rng, ei in 1:n_ens_size, ki in 1:n_k_iter
                os = os_all[ri, ei, ki, :, :]   # (n_ps, n_out)
                all(isnan.(os)) && continue
                if do_white
                    size(os, 2) != size(V_R, 1) && continue   # dim mismatch; leave cov_arr[ri,ei,ki,:] as NaN
                    sw        = Matrix((V_R' * Matrix(os')) ./ sqrt.(λ_R))'   # (n_ps, k_R)
                    truth_vec = yw
                    n_cov_dim = k_R
                else
                    sw        = os
                    truth_vec = y_truth
                    n_cov_dim = actual_out_dim
                end
                for (qi, qp) in enumerate(calibration_check_quantiles)
                    cov_arr[ri, ei, ki, qi] = mean(truth_vec[d] <= quantile(sw[:, d], qp) for d in 1:n_cov_dim)
                end
            end

            println("\n" * "="^72)
            println("Output-space marginal coverage fractions  [$(label)]")
            if do_white
                println("  x̃_d = (Vᵀx)_d / √λ_d  (R = VΛVᵀ, top $(k_R)/$(n_out) modes, $(round(Int, 100*R_variance_retain))% var threshold)")
                println("  Coverage evaluated over effective dimension = $(k_R)  (full output dim = $(n_out))")
            end
            println("  (dims as trials; expected mean_coverage ≈ q_prob for calibrated posterior)")
            println("="^72)
            println("Pooled mean coverage across all experiments:")
            for (qi, qp) in enumerate(calibration_check_quantiles)
                pool = collect(filter(!isnan, vec(cov_arr[:, :, :, qi])))
                isempty(pool) && continue
                println(@sprintf("  q = %.2f  mean = %.3f  (expected %.3f)", qp, mean(pool), qp))
            end

            local df_cov = DataFrame(
                q_prob=Float64[], ens_size=Int[], k_iter=Int[],
                mean_coverage=Union{Float64,Missing}[], std_coverage=Union{Float64,Missing}[], n_valid=Int[],
            )
            for (qi, qp) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
                vals = filter(!isnan, vec(cov_arr[:, ei, ki, qi]))
                push!(df_cov, (q_prob=qp, ens_size=ev, k_iter=kv,
                    mean_coverage = isempty(vals) ? missing : mean(vals),
                    std_coverage  = length(vals) > 1 ? std(vals) : missing,
                    n_valid       = length(vals)))
            end

            println("\nOutput-space coverage by ens_size and k_iter:")
            for ens in filter(show_ens, sort(unique(df_cov.ens_size)))
                println("\n=== ens_size = $(ens) ===")
                for (qi, qp) in enumerate(calibration_check_quantiles)
                    sub = sort(filter(r -> r.ens_size == ens && r.q_prob == qp, df_cov), :k_iter)
                    println("  q = $(qp)  (expected ≈ $(qp))")
                    show(stdout, select(sub, :k_iter, :mean_coverage, :std_coverage, :n_valid), allrows=true)
                    println()
                end
            end
        end
    end

    # ── Forcing-space coverage ─────────────────────────────────────────────
    if show_space(:forcing) && haskey(ncd, "forcing_samples") && haskey(ncd, "truth_forcing")

        n_for       = ncd.dim["forcing_dim"]
        fs_all      = coalesce.(Array(ncd["forcing_samples"]), NaN)   # (n_rng, n_ens_size, n_k_iter, n_ps, n_for)
        tf_all      = coalesce.(Array(ncd["truth_forcing"]), NaN)     # (n_rng, n_ens_size, n_for)

        cov_arr = fill(NaN, n_rng, n_ens_size, n_k_iter, length(calibration_check_quantiles))
        for ri in 1:n_rng, ei in 1:n_ens_size, ki in 1:n_k_iter
            fs      = fs_all[ri, ei, ki, :, :]
            truth_f = tf_all[ri, ei, :]
            all(isnan.(fs)) && continue
            for (qi, qp) in enumerate(calibration_check_quantiles)
                cov_arr[ri, ei, ki, qi] = mean(truth_f[d] <= quantile(fs[:, d], qp) for d in 1:n_for)
            end
        end

        println("\n" * "="^72)
        println("Forcing-space marginal coverage fractions  [raw]")
        println("  (dims as trials; expected mean_coverage ≈ q_prob for calibrated posterior)")
        println("="^72)
        println("Pooled mean coverage across all experiments:")
        for (qi, qp) in enumerate(calibration_check_quantiles)
            pool = collect(filter(!isnan, vec(cov_arr[:, :, :, qi])))
            isempty(pool) && continue
            println(@sprintf("  q = %.2f  mean = %.3f  (expected %.3f)", qp, mean(pool), qp))
        end

        df_cov = DataFrame(
            q_prob=Float64[], ens_size=Int[], k_iter=Int[],
            mean_coverage=Union{Float64,Missing}[], std_coverage=Union{Float64,Missing}[], n_valid=Int[],
        )
        for (qi, qp) in enumerate(calibration_check_quantiles), (ei, ev) in enumerate(ens_vals), (ki, kv) in enumerate(kiter_vals)
            vals = filter(!isnan, vec(cov_arr[:, ei, ki, qi]))
            push!(df_cov, (q_prob=qp, ens_size=ev, k_iter=kv,
                mean_coverage = isempty(vals) ? missing : mean(vals),
                std_coverage  = length(vals) > 1 ? std(vals) : missing,
                n_valid       = length(vals)))
        end

        println("\nForcing-space coverage by ens_size and k_iter:")
        for ens in filter(show_ens, sort(unique(df_cov.ens_size)))
            println("\n=== ens_size = $(ens) ===")
            for (qi, qp) in enumerate(calibration_check_quantiles)
                sub = sort(filter(r -> r.ens_size == ens && r.q_prob == qp, df_cov), :k_iter)
                println("  q = $(qp)  (expected ≈ $(qp))")
                show(stdout, select(sub, :k_iter, :mean_coverage, :std_coverage, :n_valid), allrows=true)
                println()
            end
        end
    end
end
