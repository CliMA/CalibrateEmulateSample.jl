using NCDatasets
using Distributions
using Statistics
using Printf
using DataFrames

indir = joinpath("hpc-variant","output","from-hpc_2026-05-29")
#filename = "ces-eki-dmc_l63_ensemble_results_2026-05-29.nc"
#filename = "ces-eki-dmc_l96_ensemble_results_2026-05-29.nc"
#filename = "ces-eki-dmc_l96_spatial_forcing_ensemble_results_2026-05-29.nc"
filename = "ces-eki-dmc_l96_nn_forcing_ensemble_results_2026-05-29.nc"

filename = joinpath(indir, filename)
@info "computing leaderboard metrics from $(filename)"

ncd = NCDataset(filename)
mh = ncd[:mahalanobis]
lp = ncd[:posterior_logpdf_true_v_map]
# Gaussian: mh = -2lp
(n_rng, n_ens_size, n_k_iter, n_par) = (ncd.dim[k] for k in ["random_seed","ensemble_size", "k_iter", "param_dim"])

mh_nonmiss = sum(.!ismissing.(mh), dims=1)[1,:,:]
lp_nonmiss = sum(.!ismissing.(lp), dims=1)[1,:,:]

qq_probs = [0.1, 0.5, 0.9]
qq_good  = quantile(Chisq(n_par), qq_probs)

# === 1. Validate: pooled empirical quantiles should match chi-sq(n_par) ===
# If mh is correctly computed it follows chi-sq(n_par), so empirical ≈ theoretical.
mh_pool = collect(skipmissing(vec(Array(mh))))
lp_pool = collect(skipmissing(vec(-2 .* Array(lp))))

mh_emp_q = quantile(mh_pool, qq_probs)
lp_emp_q = quantile(lp_pool, qq_probs)

println("Metric calibration check — pooled empirical quantiles vs chi-sq($(n_par)) (should match if metric is correct):")
println(@sprintf("  %-8s  %-12s  %-12s  %-12s", "prob", "chi-sq ref", "mh", "-2*lp"))
for (i, p) in enumerate(qq_probs)
    println(@sprintf("  %-8.2f  %-12.3f  %-12.3f  %-12.3f", p, qq_good[i], mh_emp_q[i], lp_emp_q[i]))
end

# === 2. Per-experiment scores ===
# missing where no valid seeds; * in printout where coverage < n_rng (less trusted)
mh_scores = Array{Union{Float64,Missing}}(missing, length(qq_probs), n_ens_size, n_k_iter)
lp_scores = Array{Union{Float64,Missing}}(missing, length(qq_probs), n_ens_size, n_k_iter)

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

for (qi, p) in enumerate(qq_probs)
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
for ens in sort(unique(df_scores.ens_size))
    println("\n=== ens_size = $(ens) ===")
    for (qi, p) in enumerate(qq_probs)
        sub = sort(filter(r -> r.ens_size == ens && r.q_prob == p, df_scores), :k_iter)
        println("  q = $(p)  [chi-sq($(n_par)) bound = $(round(qq_good[qi], digits=2))]")
        show(stdout, select(sub, :k_iter, :mh_score, :lp_score, :mh_coverage), allrows=true)
        println()
    end
end

