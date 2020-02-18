module EKS

using Random
using Statistics
using Sundials # CVODE_BDF() solver for ODE
using Distributions
using LinearAlgebra
using DifferentialEquations
using DocStringExtensions

export EKSObj
export construct_initial_ensemble
export compute_error
export update_ensemble!

include("EKS_bkp.jl")

"""
    EKSObj{FT<:AbstractFloat, IT<:Int}

Structure that is used in Ensemble Kalman Inversion (eks)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct EKSObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size `N_ensemble x N_parameters` containing the parameters (in each eks iteration a new array of parameters is added)"
     u::Vector{Array{FT, 2}}
     "vector of parameter names"
     unames::Vector{String}
     "vector of observations (`length: N_data`); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "vector of arrays of size `N_ensemble x N_data` containing the data ``G(u)`` (in each eks iteration a new array of data is added)"
     g::Vector{Array{FT, 2}}
     "vector of errors"
     err::Vector{FT}
     ###########################################
     # Explicit prior definition as a Gaussian #
     ###########################################
     ""
     prior_mean::Vector{FT}
     ""
     prior_cov::Array{FT, 2}
end

# outer constructors
function EKSObj(parameters::Array{FT, 2},
                parameter_names::Vector{String},
                t_mean,
                t_cov::Array{FT, 2},
                p_mean::Vector{FT},
                p_cov::Array{FT, 2}) where {FT<:AbstractFloat}

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

    prior_mean = p_mean
    prior_cov  = p_cov

    EKSObj{FT,IT}(u, parameter_names, t_mean, t_cov, N_ens, g, err, prior_mean, prior_cov)
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

function compute_error(eks)
    meang = dropdims(mean(eks.g[end], dims=1), dims=1)
    diff = eks.g_t - meang
    X = eks.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eks.err, newerr)
end


function update_ensemble!(eks::EKSObj{FT}, g) where {FT}
    # u: N_ens x N_params
    # g: N_ens x N_data
    u = eks.u[end]
    J = size(u)[1]

    u_mean = mean(u', dims = 2)
    g_mean = mean(g', dims = 2)

    g_cov = cov(g, corrected = false)
    u_cov = cov(u, corrected = false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- y
    D = 1/J * (E' * (eks.cov \ R))

    Δt = 1/maximum(real(eigvals(D)))
    noise = MvNormal(u_cov)

    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (eks.prior_cov \ u_cov')') \
                ( u' -
                    Δt * ( u' .- u_mean ) * D +
                    Δt * u_cov * (eks.prior_sigma \ eks.prior_mean)
                )

    u += implicit' + sqrt(2*Δt) * rand(noise, J)'

    # store new parameters (and observations)
    push!(eks.u, u) # N_ens x N_params
    push!(eks.g, g) # N_ens x N_data

    compute_error(eks)

end

end # module
