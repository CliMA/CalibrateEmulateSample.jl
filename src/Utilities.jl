module Utilities

using LinearAlgebra
using Statistics
using StatsBase
using Random
using ..Observations
using ..EnsembleKalmanProcesses
EnsembleKalmanProcess = EnsembleKalmanProcesses.EnsembleKalmanProcess
using ..DataContainers

export get_training_points
export get_obs_sample
export orig2zscore
export zscore2orig

"""
    extract_training_points(ekp::EnsembleKalmanProcess{FT, IT, P}, N_train_iterations::IT) where {FT,IT, P}

Extract the training points needed to train the Gaussian Process Regression.

 - `ekp` - EnsembleKalmanProcess holding the parameters and the data that were produced
             during the Ensemble Kalman (EK) process
 - `N_train_iterations` - Number of EK layers/iterations to train on

"""
function get_training_points(ekp::EnsembleKalmanProcess{FT, IT, P}, N_train_iterations::IT) where {FT, IT, P}

    # Note u[end] does not have an equivalent g
    iter_range = get_N_iterations(ekp) - N_train_iterations + 1 : get_N_iterations(ekp)

    u_tp=[]
    g_tp=[]
    for i in iter_range
        push!(u_tp, get_u(ekp, i)) #N_parameters x N_ens
        push!(g_tp, get_g(ekp, i)) #N_data x N_ens
    end
    u_tp = hcat(u_tp...) # N_parameters x (N_ek_it x N_ensemble)]
    g_tp = hcat(g_tp...) # N_data x (N_ek_it x N_ensemble)

    training_points = PairedDataContainer(u_tp,g_tp,data_are_columns=true)
    
    return training_points
end


"""
    get_obs_sample(obs::Observation; rng_seed=42, rng::Union{Random.AbstractRNG, Nothing} = nothing)

Return one random sample from the observations (for use in the MCMC)

 - `obs` - Observation struct with the observations (extract will pick one
           of the sample observations to train the j.
 - `rng_seed` - optional kwarg; seed for random number generator used to pick a 
                random sample of the observations.
- `rng` - optional kwarg; RNG object used to pick random sample. If this kwarg is
          provided, `rng_seed` is ignored.

"""
function get_obs_sample(obs::Observation; rng_seed=42, rng::Union{Random.AbstractRNG, Nothing} = nothing)
    # Ensuring reproducibility of the sampled parameter values: re-seed GLOBAL_RNG if we're
    # given a seed, but if we got an explicit rng, we shouldn't re-seed it
    rng = isnothing(rng) ? Random.seed!(rng_seed) : rng

    row_idxs = StatsBase.sample(rng, axes(obs.samples, 1), 1; replace=false, ordered=false)
    return obs.samples[row_idxs...]
end
get_obs_sample(rng::Random.AbstractRNG, obs::Observation; kwargs...) =
    get_obs_sample(obs; rng = rng, kwargs...)

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
