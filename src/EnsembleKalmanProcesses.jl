module EnsembleKalmanProcesses

#using ..Priors
using ..ParameterDistributionStorage

using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EnsembleKalmanProcess, Inversion, Sampler
export construct_initial_ensemble
export compute_error!
export update_ensemble!
export find_ekp_stepsize


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
    EnsembleKalmanProcess{FT<:AbstractFloat, IT<:Int}

Structure that is used in Ensemble Kalman processes

#Fields
$(DocStringExtensions.FIELDS)
"""
struct EnsembleKalmanProcess{FT<:AbstractFloat, IT<:Int, P<:Process}
    "vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EK iteration a new array of parameters is added)"
     u::Vector{Array{FT, 2}}
     "vector of observations (length: N_data); mean of all observation samples"
     obs_mean::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     obs_noise_cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "vector of arrays of size N_ensemble x N_data containing the model evaluations G(u) (in each EK iteration a new array of evaluations is added)"
     g::Vector{Array{FT, 2}}
     "vector of errors"
     err::Vector{FT}
     "vector of timesteps used in each EK iteration"
     Δt::Vector{FT}
     "the particular EK process (`Inversion` or `Sampler`)"
     process::P
end

# outer constructors
function EnsembleKalmanProcess(parameters::Array{FT, 2},
               obs_mean,
               obs_noise_cov::Array{FT, 2},
               process::P;
               Δt=FT(1)) where {FT<:AbstractFloat, P<:Process}

    # ensemble size
    N_ens = size(parameters)[1]
    IT = typeof(N_ens)
    # parameters
    u = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(u, parameters) # insert parameters at end of array (in this case just 1st entry)
    # G evaluations
    g = Vector{FT}[]
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    EnsembleKalmanProcess{FT, IT, P}(u, obs_mean, obs_noise_cov, N_ens, g,
                     err, Δt, process)
end


"""
construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; rng_seed=42) where {IT<:Int}

Construct the initial parameters, by sampling N_ens samples from specified
prior distribution. Returned with parameters as rows
"""
function construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; rng_seed=42) where {IT<:Int}
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    params = permutedims(sample_distribution(prior, N_ens), (2,1)) #this transpose is [N_ens x dim(param space)]
    return params
end

function compute_error!(ekp::EnsembleKalmanProcess)
    mean_g = dropdims(mean(ekp.g[end], dims=1), dims=1)
    diff = ekp.obs_mean - mean_g
    X = ekp.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ekp.err, newerr)
end


"""
   find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}
Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold.
"""
function find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # u: N_ens x N_params
    cov_init = cov(ekp.u[end], dims=1)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new=Δt)
        cov_new = cov(ekp_copy.u[end], dims=1)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt/2
        end
    end

    return Δt

end


function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g; cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}
    # u: N_ens x N_params
    u = ekp.u[end]
    cov_init = cov(ekp.u[end], dims=1)

    u_bar = fill(FT(0), size(u)[2])
    # g: N_ens x N_data
    g_bar = fill(FT(0), size(g)[2])

    cov_ug = fill(FT(0), size(u)[2], size(g)[2])
    cov_gg = fill(FT(0), size(g)[2], size(g)[2])

    if !isnothing(Δt_new)
        push!(ekp.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, FT(1))
    else
        push!(ekp.Δt, ekp.Δt[end])
    end

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

    # Update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]),
                          ekp.obs_noise_cov/ekp.Δt[end]), ekp.N_ens) # N_data x N_ens
    # Add obs_mean (N_data) to each column of noise (N_data x N_ens), then
    # transpose into N_ens x N_data
    y = (ekp.obs_mean .+ noise)'
    # N_data x N_data \ [N_ens x N_data - N_ens x N_data]'
    # --> tmp is N_data x N_ens
    tmp = (cov_gg + ekp.obs_noise_cov) \ (y - g)'
    u += (cov_ug * tmp)' # N_ens x N_params

    # store new parameters (and observations)
    push!(ekp.u, u) # N_ens x N_params
    push!(ekp.g, g) # N_ens x N_data

    compute_error!(ekp)

    # Check convergence
    cov_new = cov(ekp.u[end], dims=1)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string("New ensemble covariance determinant is less than ",
                     cov_threshold, " times its previous value.",
                     "\nConsider reducing the EK time step.")
    end
end

function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}}, g) where {FT, IT}
    # u: N_ens x N_params
    # g: N_ens x N_data
    u = ekp.u[end]
    N_ens = ekp.N_ens

    # u_mean: N_params x 1
    u_mean = mean(u', dims=2)
    # g_mean: N_params x 1
    g_mean = mean(g', dims=2)
    # g_cov: N_params x N_params
    g_cov = cov(g, corrected=false)
    # u_cov: N_params x N_params
    u_cov = cov(u, corrected=false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- ekp.obs_mean
    # D: N_ens x N_ens
    D = (1/N_ens) * (E' * (ekp.obs_noise_cov \ R))

    Δt = 1/(norm(D) + 1e-8)

    noise = MvNormal(u_cov)

    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ekp.process.prior_cov' \ u_cov')') \
                  (u'
                    .- Δt * ( u' .- u_mean) * D
                    .+ Δt * u_cov * (ekp.process.prior_cov \ ekp.process.prior_mean)
                  )

    u = implicit' + sqrt(2*Δt) * rand(noise, N_ens)'

    # store new parameters (and observations)
    push!(ekp.u, u) # N_ens x N_params
    push!(ekp.g, g) # N_ens x N_data

    compute_error!(ekp)

end

end # module EnsembleKalmanProcesses
