using NCDatasets
using JLD2
using LinearAlgebra
using Statistics
using Dates

include("experiment_config.jl")

cfg    = experiment_config(EXPERIMENT)
method = method_cases[1]

in_filename  = nc_filename(cfg, method)
out_filename = replace(in_filename, r"\.nc$" => "_minimal.nc")
prelim_filename = if cfg.force_case === nothing
    joinpath(pwd(), "output", "$(cfg.model)_computed_preliminaries.jld2")
else
    joinpath(pwd(), "output", "$(cfg.model)_computed_preliminaries_$(cfg.force_case).jld2")
end

R_variance_retain = 0.99   # fraction of R eigenvalue variance retained for whitened PCA coverage

###########################################################################
#################### Read required fields from input NC ##################
###########################################################################

@info "Reading $(in_filename)"
ncd_in = NCDataset(in_filename)

n_rng = ncd_in.dim["random_seed"]
n_ens = ncd_in.dim["ensemble_size"]
n_k   = ncd_in.dim["k_iter"]
n_cq  = ncd_in.dim["coverage_quantile"]
n_ts  = ncd_in.dim["target_scaling"]
n_out = ncd_in.dim["output_dim"]

ens_vals = Int.(ncd_in["ensemble_size"][:])
k_vals   = Int.(ncd_in["k_iter"][:])
cq_vals  = Float64.(ncd_in["coverage_quantile"][:])
ts_vals  = Float64.(ncd_in["target_scaling"][:])
os_all   = coalesce.(Array(ncd_in["output_samples"]), NaN)   # (n_rng, n_ens, n_k, n_ps, n_out)

close(ncd_in)

###########################################################################
#################### Load preliminaries and compute whitened coverage ####
###########################################################################

@info "Loading preliminaries from $(prelim_filename)"
_pd     = JLD2.load(prelim_filename)
y_truth = Float64.(_pd["y"])
R_obs   = Float64.(_pd["R"])

eig_R   = eigen(Symmetric(R_obs))
ord     = sortperm(eig_R.values; rev=true)
λ_all   = eig_R.values[ord]
V_all   = eig_R.vectors[:, ord]
cum_var = cumsum(λ_all) ./ sum(λ_all)
k_R     = something(findfirst(>=(R_variance_retain), cum_var), length(λ_all))
V_R     = V_all[:, 1:k_R]
λ_R     = λ_all[1:k_R]
yw      = V_R' * y_truth ./ sqrt.(λ_R)

@info "R-whitened PCA: retaining $(k_R)/$(n_out) modes ($(round(100*cum_var[k_R]; digits=2))% variance)"

output_coverage = fill(NaN, n_rng, n_ens, n_k, n_cq)
for ri in 1:n_rng, ei in 1:n_ens, ki in 1:n_k
    os = os_all[ri, ei, ki, :, :]
    all(isnan.(os)) && continue
    size(os, 2) != size(V_R, 1) && continue
    sw = Matrix((V_R' * Matrix(os')) ./ sqrt.(λ_R))'   # (n_ps, k_R)
    for (qi, qp) in enumerate(cq_vals)
        output_coverage[ri, ei, ki, qi] = mean(yw[d] <= quantile(sw[:, d], qp) for d in 1:k_R)
    end
end

###########################################################################
#################### Write minimal NC ####################################
###########################################################################

ds = NCDataset(out_filename, "c")

defDim(ds, "random_seed",       n_rng)
defDim(ds, "ensemble_size",     n_ens)
defDim(ds, "k_iter",            n_k)
defDim(ds, "coverage_quantile", n_cq)
defDim(ds, "target_scaling",    n_ts)
defDim(ds, "output_dim",        k_R)   # effective whitened dimension (not full output_dim = $(n_out))

ens_var = defVar(ds, "ensemble_size", Int64, ("ensemble_size",))
ens_var.attrib["description"] = "Number of ensemble members"
ens_var[:] = ens_vals

k_var = defVar(ds, "k_iter", Int64, ("k_iter",))
k_var.attrib["description"] = "Number of EKP training iterations used to fit the emulator (1-indexed)"
k_var[:] = k_vals

cq_var = defVar(ds, "coverage_quantile", Float64, ("coverage_quantile",))
cq_var.attrib["description"] = "Quantile levels used for marginal coverage fraction metrics"
cq_var[:] = cq_vals

ts_var = defVar(ds, "target_scaling", Float64, ("target_scaling",))
ts_var.attrib["description"] = "Scaling c in α_c(q) = c·√(q(1−q)/N_y); tolerance for budget_to_target / iters_to_target"
ts_var[:] = ts_vals

oc_var = defVar(ds, "output_coverage", Float64, ("random_seed", "ensemble_size", "k_iter", "coverage_quantile"); fillvalue=NaN)
oc_var.attrib["description"] = "R-whitened PCA marginal coverage: fraction of whitened output dims d where ỹ[d] ≤ q_p of whitened pushforward samples. Whitening: x̃_d = (Vᵀx)_d / √λ_d where R = VΛVᵀ. Retained $(k_R)/$(n_out) R-eigenmodes ($(round(100*cum_var[k_R]; digits=1))% variance, threshold $(R_variance_retain)). The output_dim dimension = $(k_R) is the effective whitened N_y used in α_c(q) = c·√(q(1−q)/N_y)."
oc_var[:, :, :, :] = output_coverage

close(ds)
@info "Saved minimal leaderboard data to $(out_filename)"
