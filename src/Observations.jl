module Observations

using DocStringExtensions
using LinearAlgebra
using Statistics

export Obs

"""
    Obs{FT<:AbstractFloat}

Structure that contains the observations

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Obs{FT<:AbstractFloat}
    "vector of observational samples, each of length N_data"
    samples::Vector{Vector{FT}}
    "covariance of the observational noise (assumed to be normally
     distributed); N_data x N_data. If not supplied, cov is set to a
     diagonal matrix whose non-zero elements are the sample variances"
    cov::Array{FT, 2}
    "sample mean"
    mean::Vector{FT}
    "names of the data"
    data_names::Vector{String}
end

# Constructors
function Obs(samples::Vector{Vector{FT}},
             data_names::Vector{String}) where {FT<:AbstractFloat}
    N_samples = length(samples)
    # convert to N_samples x N_data to determine sample covariance
    if N_samples == 1
        # only one sample - this requires a bit more data massaging
        temp1 = convert(Array, reshape(hcat(samples...)', N_samples, :))
        temp = dropdims(temp1, dims=1)
        samplemean = mean(temp, dims=2)
    else
        temp = convert(Array, reshape(hcat(samples...)', N_samples, :))
        samplemean = mean(temp, dims=1)
    end
    cov = cov(temp, dims=1)
    samplemean = mean(temp, dims=2)
    Obs(samples, cov, samplemean, data_names)
end

function Obs(samples::Array{FT, 2},
             data_names::Vector{String}) where {FT<:AbstractFloat}
    # samples is of size N_samples x N_data
    N_samples = size(samples, 1)
    cov = cov(samples, dims=1)
    samplemean = vec(mean(samples, dims=1))
    # convert matrix of samples to a vector of vectors
    samples = [samples[i, :] for i in 1:N_samples]
    Obs(samples, cov, samplemean, data_names)
end

function Obs(samples::Vector{Vector{FT}},
             cov::Array{FT, 2},
             data_names::Vector{String}) where {FT<:AbstractFloat}
    N_samples = length(samples)
    # convert to N_samples x N_data to determine sample covariance
    if N_samples == 1
        # only one sample - this requires a bit more data massaging
        temp1 = convert(Array, reshape(hcat(samples...)', N_samples, :))
        temp = dropdims(temp1, dims=1)
        samplemean = mean(temp, dims=2)
    else
        temp = convert(Array, reshape(hcat(samples...)', N_samples, :))
        samplemean = mean(temp, dims=1)
    end
    Obs(samples, cov, samplemean, data_names)
end

function Obs(samples::Array{FT, 2},
             cov::Array{FT, 2},
             data_names::Vector{String}) where {FT<:AbstractFloat}
    # samples is of size N_samples x N_data
    N_samples = size(samples, 1)
    samplemean = vec(mean(samples, dims=1))
    # convert matrix of samples to a vector of vectors
    samples = [samples[i, :] for i in 1:N_samples]
    Obs(samples, cov, samplemean, data_names)
end

end # module Observations
