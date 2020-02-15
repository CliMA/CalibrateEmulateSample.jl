module Utilities

using LinearAlgebra
using Statistics
using Random
using ..Observations
using ..EKI

export extract_GP_tp
export extract_obs_data
export orig2zscore
export zscore2orig
export RPAD, name, warn


"""
    extract_GP_tp(ekiobj::EKIObj{FT,IT}, N_eki_it::IT) where {FT,IT}

Extract the training points needed to train the Gaussian Process Regression.

 - `ekiobj` - EKIObj holding the parameters and the data that were produced
              during EKI
 - `N_eki_iter` - Number of EKI layers/iterations to train on

"""
function extract_GP_tp(ekiobj::EKIObj{FT,IT}, N_eki_it::IT) where {FT,IT}

    # Note u[end] does not have an equivalent g
    u_tp = ekiobj.u[end-N_eki_it:end-1] # N_eki_it x [N_ensemble x N_parameters]
    g_tp = ekiobj.g[end-N_eki_it+1:end] # N_eki_it x [N_ensemble x N_data]

    # u does not require reduction, g does:
    # g_tp[j] is jth iteration of ensembles
    u_tp = cat(u_tp..., dims=1) # [(N_eki_it x N_ensemble) x N_parameters]
    g_tp = cat(g_tp..., dims=1) # [(N_eki_it x N_ensemble) x N_data]

    return u_tp, g_tp
end


"""
    get_obs_sample(obs::Obs; rng_seed=42)

Return a random sample from the observations (for use in the MCMC)

 - `obs` - Obs struct with the observations (extract will pick one
           of the sample observations to train the j
 - `rng_seed` - seed for random number generator used to pick a random
                sample of the observations

"""
function get_obs_sample(obs::Obs; rng_seed=42)
    sample_ind = randperm!(collect(1:length(obs.samples)))[1]
    yt_sample = obs.samples[sample_ind]

    return yt_sample
end



function orig2zscore(X::AbstractVector{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    # Compute the z scores of a vector X using the given mean
    # and std
    Z = zeros(size(X))
    for i in 1:length(X)
        Z[i] = (X[i] - mean[i]) / std[i]
    end
    return Z
end

function orig2zscore(X::AbstractMatrix{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    # Compute the z scores of matrix X using the given mean and
    # std. Transformation is applied column-wise.
    Z = zeros(size(X))
    n_cols = size(X)[2]
    for i in 1:n_cols
        Z[:,i] = (X[:,i] .- mean[i]) ./ std[i]
    end
    return Z
end

function zscore2orig(Z::AbstractVector{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    # Transform X (a vector of z scores) back to the original
    # values
    X = zeros(size(Z))
    for i in 1:length(X)
      X[i] = Z[i] .* std[i] .+ mean[i]
    end
    return X
end

function zscore2orig(Z::AbstractMatrix{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    X = zeros(size(Z))
    # Transform X (a matrix of z scores) back to the original
    # values. Transformation is applied column-wise.
    n_cols = size(Z)[2]
    for i in 1:n_cols
        X[:,i] = Z[:,i] .* std[i] .+ mean[i]
    end
    return X
end

# COMMENT (MB, 02/01/2020): Not sure if the below functions are needed.
# They were already in ConvenienceFunctions before the major CES update.
const RPAD = 25

name(name::AbstractString) = rpad(name * ":", RPAD)

warn(name::AbstractString) = rpad("WARNING (" * name * "):", RPAD)

end # module ConvenienceFunctions
