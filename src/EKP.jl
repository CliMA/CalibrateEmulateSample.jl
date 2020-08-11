module EKP

using ..Priors
using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EKObj, Inversion, Sampler
export construct_initial_ensemble
export compute_error!
export update_ensemble!
export find_ek_step


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
    EKObj{FT<:AbstractFloat, IT<:Int}

Structure that is used in Ensemble Kalman processes

#Fields
$(DocStringExtensions.FIELDS)
"""
struct EKObj{FT<:AbstractFloat, IT<:Int, P<:Process}
    "vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EK iteration a new array of parameters is added)"
     u::Vector{Array{FT, 2}}
     "vector of parameter names"
     unames::Vector{String}
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
function EKObj(parameters::Array{FT, 2},
               parameter_names::Vector{String},
               obs_mean,
               obs_noise_cov::Array{FT, 2},
               process::P;
               Δt=FT(1)) where {FT<:AbstractFloat, P<:Process}

    # Throw an error if an attempt is made to instantiate a `Sampler` EKObj
    # EK Sampler implementation is not finalized yet, so its use is prohibited
    # TODO: Finalize EKS implementation (can be done as soon as we know the 
    #       correct EKS update equation, which apparently is different from
    #       Eq. (2.8) in Cleary et al. (2019)) 
    err_msg = "Ensemble Kalman Sampler is not fully implemented yet. Use Ensemble Kalman Inversion instead."
    typeof(process) != Sampler{FT} || error(err_msg)

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

    EKObj{FT, IT, P}(u, parameter_names, obs_mean, obs_noise_cov, N_ens, g, 
                     err, Δt, process)
end


"""
construct_initial_ensemble(N_ens::IT, priors::Array{Prior, 1}; rng_seed=42) where {IT<:Int}

Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::IT, priors::Array{Prior, 1}; rng_seed=42) where {IT<:Int}
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i].dist
        if !(typeof(priors[i].dist) == Deterministic{Float64})
            params[:, i] = rand(prior_i, N_ens)
        else
            params[:, i] = prior_i.value * ones(N_ens)
        end
    end

    return params
end

function compute_error!(ek::EKObj)
    mean_g = dropdims(mean(ek.g[end], dims=1), dims=1)
    diff = ek.obs_mean - mean_g
    X = ek.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ek.err, newerr)
end


"""
   find_ek_step(ek::EKObj{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}
Find largest step for the EK solver that leads to a reduction of the determinant of the sample  
covariance matrix no greater than cov_threshold.
"""
function find_ek_step(ek::EKObj{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT, IT}
    accept_step = false
    if !isempty(ek.Δt)
        Δt = deepcopy(ek.Δt[end])
    else
        Δt = FT(1)
    end
    # u: N_ens x N_params
    cov_init = cov(ek.u[end], dims=1)
    while accept_step == false
        ek_copy = deepcopy(ek)
        update_ensemble!(ek_copy, g, Δt_new=Δt)
        cov_new = cov(ek_copy.u[end], dims=1)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_step = true
        else
            Δt = Δt/2
        end
    end

    return Δt

end


function update_ensemble!(ek::EKObj{FT, IT, Inversion}, g; cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}
    # u: N_ens x N_params
    u = ek.u[end]
    cov_init = cov(ek.u[end], dims=1)

    u_bar = fill(FT(0), size(u)[2])
    # g: N_ens x N_data
    g_bar = fill(FT(0), size(g)[2])

    cov_ug = fill(FT(0), size(u)[2], size(g)[2])
    cov_gg = fill(FT(0), size(g)[2], size(g)[2])

    if !isnothing(Δt_new)
        push!(ek.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ek.Δt)
        push!(ek.Δt, FT(1))
    else
        push!(ek.Δt, ek.Δt[end])
    end

    # update means/covs with new param/observation pairs u, g
    for j = 1:ek.N_ens

        u_ens = u[j, :]
        g_ens = g[j, :]

        # add to mean
        u_bar += u_ens
        g_bar += g_ens

        #add to cov
        cov_ug += u_ens * g_ens' # cov_ug is N_params x N_data
        cov_gg += g_ens * g_ens'
    end

    u_bar = u_bar / ek.N_ens
    g_bar = g_bar / ek.N_ens
    cov_ug = cov_ug / ek.N_ens - u_bar * g_bar'
    cov_gg = cov_gg / ek.N_ens - g_bar * g_bar'

    # Update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]), 
                          ek.obs_noise_cov/ek.Δt[end]), ek.N_ens) # N_data x N_ens
    # Add obs_mean (N_data) to each column of noise (N_data x N_ens), then 
    # transpose into N_ens x N_data
    y = (ek.obs_mean .+ noise)' 
    # N_data x N_data \ [N_ens x N_data - N_ens x N_data]' 
    # --> tmp is N_data x N_ens
    tmp = (cov_gg + ek.obs_noise_cov) \ (y - g)' 
    u += (cov_ug * tmp)' # N_ens x N_params

    # store new parameters (and observations)
    push!(ek.u, u) # N_ens x N_params
    push!(ek.g, g) # N_ens x N_data

    compute_error!(ek)

    # Check convergence
    cov_new = cov(ek.u[end], dims=1)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string("New ensemble covariance determinant is less than ", 
                     cov_threshold, " times its previous value.",
                     "\nConsider reducing the EK time step.")
    end
end

function update_ensemble!(ek::EKObj{FT, IT, Sampler{FT}}, g) where {FT, IT}
    # u: N_ens x N_params
    # g: N_ens x N_data
    u = ek.u[end]
    N_ens = ek.N_ens

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
    R = g' .- ek.obs_mean
    # D: N_ens x N_ens
    D = 1/N_ens * (E' * (ek.obs_noise_cov \ R))

    Δt = 1/(norm(D) + 1e-8)
    noise = MvNormal(u_cov)

    ###########################################################################
    ###############    TODO: Implement correct equation here   ################
    ###########################################################################

    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ek.process.prior_cov \ u_cov)) \
                  (u' 
                    - Δt * ( u' .- u_mean) * D  
                    .+ Δt * u_cov * (ek.process.prior_cov \ ek.process.prior_mean)
                  )

    u += implicit' + sqrt(2*Δt) * rand(noise, N_ens)'

    ###########################################################################
    ###########################################################################

    # store new parameters (and observations)
    push!(ek.u, u) # N_ens x N_params
    push!(ek.g, g) # N_ens x N_data

    compute_error!(ek)

end

end # module EKP
