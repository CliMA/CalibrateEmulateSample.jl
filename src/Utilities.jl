module Utilities

using LinearAlgebra
using Statistics
using Random
using ..Observations
using ..EKP

export extract_GP_tp
export get_obs_sample
export orig2zscore
export zscore2orig

"""
    extract_GP_tp(ekobj::EKObj{FT, IT, P}, N_ek_it::IT) where {FT,IT, P}

Extract the training points needed to train the Gaussian Process Regression.

 - `ekobj` - EKObj holding the parameters and the data that were produced
             during the Ensemble Kalman (EK) process
 - `N_ek_iter` - Number of EK layers/iterations to train on

"""
function extract_GP_tp(ekobj::EKObj{FT, IT, P}, N_ek_it::IT) where {FT, IT, P}

    # Note u[end] does not have an equivalent g
    u_tp = ekobj.u[end-N_ek_it:end-1] # N_ek_it x [N_ensemble x N_parameters]
    g_tp = ekobj.g[end-N_ek_it+1:end] # N_ek_it x [N_ensemble x N_data]

    # u does not require reduction, g does:
    # g_tp[j] is jth iteration of ensembles
    u_tp = cat(u_tp..., dims=1) # [(N_ek_it x N_ensemble) x N_parameters]
    g_tp = cat(g_tp..., dims=1) # [(N_ek_it x N_ensemble) x N_data]

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

end # module
