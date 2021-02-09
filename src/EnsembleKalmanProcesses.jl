module EnsembleKalmanProcesses

#using ..Priors
using ..ParameterDistributionStorage
using ..DataStorage

using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export EnsembleKalmanProcess, Inversion, Sampler
export get_u_final, get_u, get_g, get_N_iterations
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
    "Array of stores for parameters (u), each of size [N_data x N_ensemble]"
    u::Array{DataContainer{FT}}
    "vector of the observed vector size [N_data]"
    obs_mean::Vector{FT}
    "covariance of the observational noise, which is assumed to be normally distributed"
    obs_noise_cov::Array{FT, 2}
    "ensemble size"
    N_ens::IT
    "Array of stores for forward model outputs, each of size  [N_data x N_ensembles]; mean of all observation samples"
    g::Array{DataContainer{FT}}
    "vector of errors"
    err::Vector{FT}
    "vector of timesteps used in each EK iteration"
    Δt::Vector{FT}
    "the particular EK process (`Inversion` or `Sampler`)"
    process::P
end

# outer constructors
function EnsembleKalmanProcess(params::Array{FT, 2},
                               obs_mean,
                               obs_noise_cov::Array{FT, 2},
                               process::P;
                               data_are_columns = true,
                               Δt=FT(1)) where {FT<:AbstractFloat, P<:Process}

    init_params=DataContainer(params, data_are_columns=data_are_columns)
    # ensemble size
    N_ens = size(get_data(init_params))[2] #stored with data as columns
    IT = typeof(N_ens)
    # parameters
    g=[] #populated once we have evaluations
    # error store
    err = FT[]
    # timestep store
    Δt = Array([Δt])

    EnsembleKalmanProcess{FT, IT, P}([init_params], obs_mean, obs_noise_cov, N_ens, g,
                                     err, Δt, process)
end



"""
    get_X(ekp::EnsembleKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}

Get X=u or X=g for the EKI iteration. Returns a DataContainer object unless array is specified.
"""
function get_u(ekp::EnsembleKalmanProcess, iteration::IT; return_array=true) where {IT <: Integer}
    return  return_array ? get_data(ekp.u[iteration]) : ekp.u[iteration]
end

function get_g(ekp::EnsembleKalmanProcess,iteration::IT; return_array=true) where {IT <: Integer}
    return return_array ? get_data(ekp.g[iteration]) : ekp.g[iteration]
end

"""
    get_u_X(ekp::EnsembleKalmanProcess, return_array=true)

Get the X=final or X=prior iteration of parameters (the "solution" to the optimization problem), returns a DataContainer Object if return_array is false.
"""
function get_u_final(ekp::EnsembleKalmanProcess; return_array=true)
    return return_array ? get_u(ekp,size(ekp.u)[1]) : ekp.u[end]
end

function get_u_prior(ekp::EnsembleKalmanProcess; return_array=true)
    return return_array ? get_u(ekp,1) : ekp.u[1]
end

"""
    get_N_iterations(ekp::EnsembleKalmanProcess

get number of times update has been called (equals size(g), or size(u)-1) 
"""
function get_N_iterations(ekp::EnsembleKalmanProcess)
    return size(ekp.u)[1] - 1 
end
"""
    construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; rng_seed=42) where {IT<:Int}

Construct the initial parameters, by sampling N_ens samples from specified
prior distribution. Returned with parameters as columns
"""
function construct_initial_ensemble(prior::ParameterDistribution, N_ens::IT; rng_seed=42) where {IT<:Int}
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    parameters = sample_distribution(prior, N_ens) #of size [dim(param space) N_ens]
    return parameters
end

function compute_error!(ekp::EnsembleKalmanProcess)
    mean_g = dropdims(mean(get_data(ekp.g[end]), dims=2), dims=2)
    diff = ekp.obs_mean - mean_g
    X = ekp.obs_noise_cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(ekp.err, newerr)
end


"""
   find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT, 2}, data_are_columns::Bool; cov_threshold::FT=0.01) where {FT}
Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT,2}, data_are_columns::Bool; cov_threshold::FT=0.01) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # final_params [N_params x N_ens]
    cov_init = cov(get_u_final(ekp), dims=2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, data_are_columns, Δt_new=Δt)
        cov_new = cov(get_u_final(ekp_copy), dims=2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt/2
        end
    end

    return Δt

end

"""
    update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, <:Process}, g_in::Array{FT,2}, data_are_columns::Bool, cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}

Updates the ensemble according to which type of Process we have. User must provide a boolean to indicate whether model outputs g_in are stored as columms (data_are_columns=true) or rows (false)
"""
function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g_in::Array{FT,2}, data_are_columns::Bool; cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}

    # We enforce that data are rows
    # u: N_ens x N_params
    # g: N_ens x N_data
    u_old = get_u_final(ekp)
    u_old = permutedims(u_old,(2,1))    
    u = u_old
    g = data_are_columns ? permutedims(g_in, (2,1)) : g_in
    #catch works when g_in non-square
    if !(size(g)[1] == ekp.N_ens) 
         throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try flipping the boolean data_are_columns? (default is true), or check ensemble size"))
    end
           
    cov_init = cov(u, dims=1)

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

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns=false))
    push!(ekp.g, DataContainer(g, data_are_columns=false))
    # u_old is N_ens x N_params, g is N_ens x N_data,
    # but stored in data container with N_ens as the 2nd dim
    
    compute_error!(ekp)

    # Check convergence
    cov_new = cov(get_u_final(ekp), dims=2)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string("New ensemble covariance determinant is less than ",
                     cov_threshold, " times its previous value.",
                     "\nConsider reducing the EK time step.")
    end
end

function update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, Sampler{FT}}, g_in::Array{FT,2}, data_are_columns::Bool) where {FT, IT}

    # u: N_ens x N_params
    # g: N_ens x N_data
    u_old = get_u_final(ekp)
    u_old = permutedims(u_old,(2,1))
    u = u_old
    g = data_are_columns ? permutedims(g_in, (2,1)) : g_in
    #catch works when g_in non-square 
    if !(size(g)[1] == ekp.N_ens) 
         throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g_in do not match, try flipping the boolean data_are_columns? (default is true), or check ensemble size"))
    end

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
    D = (1/ekp.N_ens) * (E' * (ekp.obs_noise_cov \ R))

    Δt = 1/(norm(D) + 1e-8)

    noise = MvNormal(u_cov)

    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (ekp.process.prior_cov' \ u_cov')') \
                  (u'
                    .- Δt * ( u' .- u_mean) * D
                    .+ Δt * u_cov * (ekp.process.prior_cov \ ekp.process.prior_mean)
                  )

    u = implicit' + sqrt(2*Δt) * rand(noise, ekp.N_ens)'

    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns=false))
    push!(ekp.g, DataContainer(g, data_are_columns=false))
    # u_old is N_ens x N_params, g is N_ens x N_data,
    # but stored in data container with N_ens as the 2nd dim

    compute_error!(ekp)

end

end # module EnsembleKalmanProcesses
