module EnsembleKalmanProcesses

using Random
using Statistics
using Sundials # CVODE_BDF() solver for ODE
using Distributions
using LinearAlgebra
using DifferentialEquations
using DocStringExtensions

export EnsembleKalmanProcess, Inversion, Sampler
export construct_initial_ensemble
export compute_error
export update_ensemble!

abstract type Process end

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion <: Process end

"""
    Sampler{FT<:AbstractFloat,IT<:Int} <: Process

An ensemble Kalman Sampler process
"""
struct Sampler{FT<:AbstractFloat} <: Process
  ""
  prior_mean::Vector{FT}
  ""
  prior_cov::Array{FT, 2}
end

"""
    EnsembleKalmanProcess{FT<:AbstractFloat,IT<:Int}

Structure that is used in Ensemble Kalman Inversion (EKI)

#Fields
$(DocStringExtensions.FIELDS)
"""
struct EnsembleKalmanProcess{FT<:AbstractFloat,IT<:Int,P<:Process}
    "a vector of arrays of size `N_ensemble x N_parameters` containing the parameters (in each EKI iteration a new array of parameters is added)"
     u::Vector{Array{FT, 2}}
     "vector of parameter names"
     unames::Vector{String}
     "vector of observations `(length: N_data)`; mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "vector of arrays of size `N_ensemble x N_data` containing the data ``G(u)`` (in each EKI iteration a new array of data is added)"
     g::Vector{Array{FT, 2}}
     "vector of errors"
     err::Vector{FT}
     "the particular EK process (`Inversion` or `Sampler`)"
     process::P
end

"""
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}

Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i]
        params[:, i] = rand(prior_i, N_ens)
    end

    return params
end

function compute_error!(ekp::EnsembleKalmanProcess)
    meang = dropdims(mean(ekp.g[end], dims=1), dims=1)
    diff = ekp.g_t - meang
    X = ekp.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ekp.err, newerr)
end

function EnsembleKalmanProcess(parameters::Array{FT, 2},
                               parameter_names::Vector{String},
                               t_mean,
                               t_cov::Array{FT, 2},
                               process::P=Inversion()) where {FT<:AbstractFloat,P<:Process}

    # ensemble size
    N_ens = size(parameters)[1]
    IT = typeof(N_ens)
    # parameters
    u = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(u, parameters) # insert parameters at end of array (in this case just 1st entry)
    # observations
    g = Vector{FT}[]
    # error store
    err = []

    EnsembleKalmanProcess{FT,IT,P}(u, parameter_names, t_mean, t_cov, N_ens, g, err, process)
end

function update_ensemble!(ekp::EnsembleKalmanProcess{FT,IT,Inversion}, g) where {FT,IT}
    # u: N_ens x N_params
    u = ekp.u[end]

    u_bar = fill(FT(0), size(u)[2])
    # g: N_ens x N_data
    g_bar = fill(FT(0), size(g)[2])

    cov_ug = fill(FT(0), size(u)[2], size(g)[2])
    cov_gg = fill(FT(0), size(g)[2], size(g)[2])

    # update means/covs with new param/observation pairs u, g
    for j = 1:ekp.N_ens

        u_ens = u[j, :]
        g_ens = g[j, :]

        # add to mean
        u_bar += u_ens
        g_bar += g_ens

        #add to cov
        cov_ug += u_ens * g_ens' # cov_ug is N_params x N_data
        cov_gg += g_ens * g_ens'
    end

    u_bar = u_bar / ekp.N_ens
    g_bar = g_bar / ekp.N_ens
    cov_ug = cov_ug / ekp.N_ens - u_bar * g_bar'
    cov_gg = cov_gg / ekp.N_ens - g_bar * g_bar'

    # update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]), ekp.cov), ekp.N_ens) # N_data * N_ens
    y = (ekp.g_t .+ noise)' # add g_t (N_data) to each column of noise (N_data x N_ens), then transp. into N_ens x N_data
    tmp = (cov_gg + ekp.cov) \ (y - g)' # N_data x N_data \ [N_ens x N_data - N_ens x N_data]' --> tmp is N_data x N_ens
    u += (cov_ug * tmp)' # N_ens x N_params

    # store new parameters (and observations)
    push!(ekp.u, u) # N_ens x N_params
    push!(ekp.g, g) # N_ens x N_data

    compute_error!(ekp)

end

function update_ensemble!(ekp::EnsembleKalmanProcess{FT,IT,Sampler}, g) where {FT,IT}
    # u: N_ens x N_params
    # g: N_ens x N_data
    u = ekp.u[end]
    J = size(u)[1]

    u_mean = mean(u', dims = 2)
    g_mean = mean(g', dims = 2)

    g_cov = cov(g, corrected = false)
    u_cov = cov(u, corrected = false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- y
    D = 1/J * (E' * (ekp.cov \ R))

    Δt = 1/(norm(D) + 1e-8)
    noise = MvNormal(u_cov)

    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ekp.process.prior_cov \ u_cov')') \
                ( u' -
                    Δt * ( u' .- u_mean ) * D +
                    Δt * u_cov * (ekp.prior_sigma \ ekp.process.prior_mean)
                )

    u += implicit' + sqrt(2*Δt) * rand(noise, J)'

    # store new parameters (and observations)
    push!(ekp.u, u) # N_ens x N_params
    push!(ekp.g, g) # N_ens x N_data

    compute_error!(ekp)

end

end # module EKI
