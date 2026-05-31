using NCDatasets
using Distributions
using Statistics

#filename = "ces-eki-dmc_l63_ensemble_results_2026-05-29.nc"
#filename = "ces-eki-dmc_l96_ensemble_results_2026-05-29.nc"
#filename = "ces-eki-dmc_l96_spatial_forcing_ensemble_results_2026-05-29.nc"
filename = "ces-eki-dmc_l96_nn_forcing_ensemble_results_2026-05-29.nc"

# load
ncd = NCDataset(filename)
mh = ncd[:mahalanobis]
lp = ncd[:posterior_logpdf_true_v_map]
# Gaussian: mh=-2lp
(n_rng, n_ens_size, n_k_iter, n_par) = (ncd.dim[k] for k in ["random_seed","ensemble_size", "k_iter", "param_dim"])
# assume dimnames(mh) = "random_seed", "ensemble_size", "k_iter"

# count missings
mh1_nonmiss = sum(.!ismissing.(mh), dims=1)[1,:,:]
lp1_nonmiss = sum(.!ismissing.(lp), dims=1)[1,:,:]

# criteria of "good" posterior
qq_good = quantile(Chisq(n_par), [0.1, 0.5, 0.9])
# now we compare over the rng dimension
mh_scores = fill(NaN, length(qq_good), size(mh,2), size(mh,3))
lp_scores = fill(NaN, length(qq_good), size(lp,2), size(lp,3))
for (idx,qg) in enumerate(qq_good)
    mh_scores[idx,:,:] = [sum(skipmissing(mh[:,i,j]) .<= qg; init=0) for i in axes(mh,2), j in axes(mh,3)]
    lp_scores[idx,:,:] = [sum(skipmissing(-2*lp[:,i,j]) .<= qg; init=0) for i in axes(lp,2), j in axes(lp,3)]
    # divide by the # non-missing values. In case of all missing we divide by 1
    mh_scores[idx,:,:] ./= max.(mh1_nonmiss,1)
    lp_scores[idx,:,:] ./= max.(lp1_nonmiss,1)
end


