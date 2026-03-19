# refactor-Sample
module MarkovChainMonteCarlo

using ..Emulators
import ..Emulators: get_encoder_schedule
using ..Utilities

using ..ParameterDistributions
using ..EnsembleKalmanProcesses

import Distributions: sample # Reexport sample()
using Distributions
using DocStringExtensions
using LinearAlgebra
using Printf
using Random
using Statistics
using ForwardDiff
using ReverseDiff

using MCMCChains
import AbstractMCMC: sample # Reexport sample()
using AbstractMCMC
import AdvancedMH

export EmulatorPosteriorModel,
    MetropolisHastingsSampler,
    MCMCProtocol,
    GradFreeProtocol,
    ForwardDiffProtocol,
    ReverseDiffProtocol,
    RWMHSampling,
    pCNMHSampling,
    BarkerSampling,
    MCMCWrapper,
    accept_ratio,
    optimize_stepsize,
    get_posterior,
    get_sample_kwargs,
    get_encoder_schedule,
    sample,
    esjd

    # ------------------------------------------------------------------------------------------
    # Sampler extensions to differentiate vanilla RW and pCN algorithms
    #
    # (Strictly speaking the difference between RW and pCN should be implemented at the level of
    # the MH Sampler's Proposal, not by defining a new Sampler, since the former is where the 
    # only change is made. We do the latter here because doing the former would require more
    # boilerplate code (repeating AdvancedMH/src/proposal.jl for the new Proposals)).

    # some possible types of autodiff used

"""
$(DocStringExtensions.TYPEDEF)

Type used to dispatch different autodifferentiation methods where different emulators have a different compatability with autodiff packages 
"""
abstract type AutodiffProtocol end

"""
$(DocStringExtensions.TYPEDEF)

Type to construct samplers for emulators not compatible with autodifferentiation
"""
abstract type GradFreeProtocol <: AutodiffProtocol end

"""
$(DocStringExtensions.TYPEDEF)

Type to construct samplers for emulators compatible with `ForwardDiff.jl` autodifferentiation
"""
abstract type ForwardDiffProtocol <: AutodiffProtocol end

"""
$(DocStringExtensions.TYPEDEF)

Type to construct samplers for emulators compatible with `ReverseDiff.jl` autodifferentiation
"""
abstract type ReverseDiffProtocol <: AutodiffProtocol end
# ...to be implemented...
#=
abstract type ZygoteProtocol <: AutodiffProtocol end
abstract type EnzymeProtocol <: AutodiffProtocol end
=#

function _get_proposal(prior::ParameterDistribution)
    # *only* use covariance of prior, not full distribution
    Σsqrt = sqrt(ParameterDistributions.cov(prior)) # rt_cov * MVN(0,I) avoids the posdef errors for MVN in Julia Distributions   
    return AdvancedMH.RandomWalkProposal(Σsqrt * MvNormal(zeros(size(Σsqrt)[1]), I))
end


"""
$(DocStringExtensions.TYPEDEF)

Type used to dispatch different methods of the [`MetropolisHastingsSampler`](@ref) 
constructor, corresponding to different sampling algorithms.
"""
abstract type MCMCProtocol end

"""
$(DocStringExtensions.TYPEDEF)
    
[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for 
new parameters as as vanilla random walk, based on the covariance of `prior`.
"""
struct RWMHSampling{T <: AutodiffProtocol} <: MCMCProtocol end

RWMHSampling() = RWMHSampling{GradFreeProtocol}()

struct RWMetropolisHastings{PT, ADT <: AutodiffProtocol} <: AdvancedMH.MHSampler
    proposal::PT
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::RWMetropolisHastings,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructor for all `Sampler` objects, with one method for each supported MCMC algorithm.

!!! warning
    Both currently implemented Samplers use a Gaussian approximation to the prior: 
    specifically, new Metropolis-Hastings proposals are a scaled combination of the old 
    parameter values and a random shift distributed as a Gaussian with the same covariance
    as the `prior`. 
    
    This suffices for our current use case, in which our priors are Gaussian
    (possibly in a transformed space) and we assume enough fidelity in the Emulator that 
    inference isn't prior-dominated.
"""
function MetropolisHastingsSampler(::RWMHSampling{T}, prior::ParameterDistribution) where {T <: AutodiffProtocol}
    proposal = _get_proposal(prior)
    return RWMetropolisHastings{typeof(proposal), T}(proposal)
end

"""
$(DocStringExtensions.TYPEDEF)
    
[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for 
new parameters according to the preconditioned Crank-Nicholson (pCN) algorithm, which is 
usable for MCMC in the *stepsize → 0* limit, unlike the vanilla random walk. Steps are based 
on the covariance of `prior`.
"""
struct pCNMHSampling{T <: AutodiffProtocol} <: MCMCProtocol end
pCNMHSampling() = pCNMHSampling{GradFreeProtocol}()

struct pCNMetropolisHastings{D, T <: AutodiffProtocol} <: AdvancedMH.MHSampler
    proposal::D
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::pCNMetropolisHastings,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)
function MetropolisHastingsSampler(::pCNMHSampling{T}, prior::ParameterDistribution) where {T <: AutodiffProtocol}
    proposal = _get_proposal(prior)
    return pCNMetropolisHastings{typeof(proposal), T}(proposal)
end

#------ The following are gradient-based samplers

"""
$(DocStringExtensions.TYPEDEF)

[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for
new parameters according to the Barker proposal.
"""
struct BarkerSampling{T <: AutodiffProtocol} <: MCMCProtocol end
BarkerSampling() = BarkerSampling{ForwardDiffProtocol}()

struct BarkerMetropolisHastings{D, T <: AutodiffProtocol} <: AdvancedMH.MHSampler
    proposal::D
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::BarkerMetropolisHastings,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)

function MetropolisHastingsSampler(::BarkerSampling{T}, prior::ParameterDistribution) where {T <: AutodiffProtocol}
    proposal = _get_proposal(prior)
    return BarkerMetropolisHastings{typeof(proposal), T}(proposal)
end


## -------- Autodifferentiation procedures ------ ##

function autodiff_gradient(model::AdvancedMH.DensityModel, params, autodiff_protocol)
    if autodiff_protocol == ForwardDiffProtocol
        return ForwardDiff.gradient(x -> AdvancedMH.logdensity(model, x), params)
    elseif autodiff_protocol == ReverseDiffProtocol
        return ReverseDiff.gradient(x -> AdvancedMH.logdensity(model, x), params)
    else
        throw(
            ArgumentError(
                "Calling `autodiff_gradient(...)` on a sampler with protocol $(autodiff_protocol) that has *no* gradient implementation.\n Please select from a protocol with a gradient implementation (e.g., `ForwardDiffProtocol`).",
            ),
        )
    end

end

autodiff_gradient(model::AdvancedMH.DensityModel, params, sampler::MH) where {MH <: AdvancedMH.MHSampler} =
    autodiff_gradient(model::AdvancedMH.DensityModel, params, typeof(sampler).parameters[2]) # hacky way of getting the "AutodiffProtocol"

function autodiff_hessian(model::AdvancedMH.DensityModel, params, autodiff_protocol)
    if autodiff_protocol == ForwardDiffProtocol
        return Symmetric(ForwardDiff.hessian(x -> AdvancedMH.logdensity(model, x), params))
    elseif autodiff_protocol == ReverseDiffProtocol
        return Symmetric(ReverseDiff.hessian(x -> AdvancedMH.logdensity(model, x), params))
    else
        throw(
            ArgumentError(
                "Calling `autodiff_hessian(...)` on a sampler with protocol $(autodiff_protocol) that has *no* hessian implementation.\n Please select from a protocol with a hessian implementation (e.g., `ForwardDiffProtocol`).",
            ),
        )

    end
end

autodiff_hessian(model::AdvancedMH.DensityModel, params, sampler::MH) where {MH <: AdvancedMH.MHSampler} =
    autodiff_hessian(model, params, typeof(sampler).parameters[2])




# ------------------------------------------------------------------------------------------
# Use emulated model in sampler

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Defines the internal log-density function over a vector of observation samples using an assumed conditionally indepedent likelihood, that is with a log-likelihood of `ℓ(y,θ) = sum^n_i log( p(y_i|θ) )`.

Inputs:
=======
- θ: Parameters in physical (constrained) coordinates.
- prior: Prior distribution as a ParameterDistribution
- em: Emulator object with predict(.) method, evaluations returned in encoded space 
- obs_vec: encoded data vector sample(s)
"""
function emulator_log_density_model(
    θ,
    prior::ParameterDistribution,
    em_or_fmw::EorFMW,
    obs_vec::AV,
) where {AV <: AbstractVector, EorFMW <: Union{Emulator, ForwardMapWrapper}}

    # 
    # transform_to_real = false means g, g_cov, obs_sample are in encoded coords.

    # predict is written to apply to columns.
    # Returned g is a length-1, Vector{Real} or Vector{Vector}, and g_cov is length-1 Vector{Vector} or Vector{Matrix} respectively
    g, g_cov = Emulators.predict(em_or_fmw, reshape(θ, :, 1), encode = "in_and_out", add_obs_noise_cov = true)

    decoded_θ = vec(decode_data(em_or_fmw, reshape(θ, :, 1), "in")) # we still compute logpdf(prior, decoded_θ) in full space

    if isa(g_cov[1], Real)
        return sum([logpdf(MvNormal(obs, g_cov[1] * I), vec(g)) for obs in obs_vec]) + logpdf(prior, decoded_θ)
    else
        return sum([logpdf(MvNormal(obs, g_cov[1]), vec(g)) for obs in obs_vec]) + logpdf(prior, decoded_θ)
    end

end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Factory which constructs `AdvancedMH.DensityModel` objects given a prior on the model 
parameters (`prior`) and an [`Emulator`](@ref) of the log-likelihood of the data given 
parameters. Together these yield the log posterior density we're attempting to sample from 
with the MCMC, which is the role of the `DensityModel` class in the `AbstractMCMC` interface.
"""
function EmulatorPosteriorModel(
    prior::ParameterDistribution,
    em_or_fmw::EorFMW,
    obs_vec::AV,
) where {AV <: AbstractVector, EorFMW <: Union{Emulator, ForwardMapWrapper}}

    return AdvancedMH.DensityModel(x -> emulator_log_density_model(x, prior, em_or_fmw, obs_vec))
end

# ------------------------------------------------------------------------------------------
# Record MH accept/reject decision in MCMCState object

"""
$(DocStringExtensions.TYPEDEF)

Extends the `AdvancedMH.Transition` (which encodes the current state of the MC during
sampling) with a boolean flag to record whether this state is new (arising from accepting a
Metropolis-Hastings proposal) or old (from rejecting a proposal).

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
struct MCMCState{T, L <: Real} <: AdvancedMH.AbstractTransition
    "Sampled value of the parameters at the current state of the MCMC chain."
    params::T
    "Log probability of `params`, as computed by the model using the prior."
    log_density::L
    "Whether this state resulted from accepting a new MH proposal."
    accepted::Bool
end

# Boilerplate from AdvancedMH:
# Store the new draw and its log density.
MCMCState(model::AdvancedMH.DensityModel, params, accepted = true) =
    MCMCState(params, logdensity(model, params), accepted)

AdvancedMH.logdensity(model::AdvancedMH.DensityModel, t::MCMCState) = t.log_density

# AdvancedMH.transition() is only called to create a new proposal, so create a MCMCState
# with accepted = true since that object will only be used if proposal is accepted.
function AdvancedMH.transition(
    sampler::MHS,
    model::AdvancedMH.DensityModel,
    params,
    log_density::Real,
) where {MHS <: Union{pCNMetropolisHastings, RWMetropolisHastings, BarkerMetropolisHastings}}
    return MCMCState(params, log_density, true)
end

# method extending AdvancedMH.propose() to vanilla random walk with explicitly given stepsize
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::RWMetropolisHastings,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    return current_state.params + stepsize * rand(rng, sampler.proposal)
end

# method extending AdvancedMH.propose() for preconditioned Crank-Nicholson
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::pCNMetropolisHastings,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    # Use prescription in Beskos et al (2017) "Geometric MCMC for infinite-dimensional 
    # inverse problems." for relating ρ to Euler stepsize:
    ρ = (1 - stepsize / 4) / (1 + stepsize / 4)
    return ρ * current_state.params .+ sqrt(1 - ρ^2) * rand(rng, sampler.proposal)
end

# method extending AdvancedMH.propose() for the Barker proposal
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::BarkerMetropolisHastings,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    # Livingstone and Zanella (2022)
    # Compute the gradient of the log-density at the current state
    n = length(current_state.params)
    log_gradient = autodiff_gradient(model, current_state.params, sampler)
    xi = rand(rng, sampler.proposal)
    return current_state.params .+ (stepsize .* ((rand(rng, n) .< 1 ./ (1 .+ exp.(-log_gradient .* xi))) .* xi))
end

# Copy a MCMCState and set accepted = false
reject_transition(t::MCMCState) = MCMCState(t.params, t.log_density, false)

# Metropolis-Hastings logic. We need to add 2 things to step() implementation in AdvancedMH:
# 1) stepsize-dependent propose(); 2) record whether proposal accepted/rejected in MCMCState
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.MHSampler,
    current_state::MCMCState;
    stepsize::FT = 1.0,
    kwargs...,
) where {FT <: AbstractFloat}
    # Generate a new proposal.
    new_params = AdvancedMH.propose(rng, sampler, model, current_state; stepsize = stepsize)
    # Calculate the log acceptance probability and the log density of the candidate.
    new_log_density = AdvancedMH.logdensity(model, new_params)

    # Just to initialize: if you pass state, it just reads the old log_density (initialized as false). in this case, compute it by passing the actual parameter values
    current_log_density =
        isa(AdvancedMH.logdensity(model, current_state), Bool) ? AdvancedMH.logdensity(model, current_state.params) :
        AdvancedMH.logdensity(model, current_state)

    log_α =
        new_log_density - current_log_density + AdvancedMH.logratio_proposal_density(sampler, current_state, new_params)

    # Decide whether to return the previous params or the new one.
    new_state = if -Random.randexp(rng) < log_α
        # accept
        AdvancedMH.transition(sampler, model, new_params, new_log_density)
    else
        # reject
        reject_transition(current_state)
    end
    # Return a 2-tuple consisting of the next sample and the the next state.
    # In this case (MH obeying detailed balance) they are identical.
    return new_state, new_state
end

# ------------------------------------------------------------------------------------------
# Extend the record-keeping methods defined in AdvancedMH to include the 
# MCMCState.accepted field added above.

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:MCMCState},
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.MHSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial = 0,
    thinning = 1,
    param_names = missing,
    kwargs...,
)
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.params, t.log_density, t.accepted) for t in ts]
    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].params))]
        #    elseif length(param_names) < length(keys(ts[1].params))# in case bug with MV names, Chains still needs one name per dist.
        #        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end
    internal_names = [:log_density, :accepted]

    # Bundle everything up and return a MCChains.Chains struct.
    return MCMCChains.Chains(
        vals, # current state information as vec-of-vecs
        vcat(param_names, internal_names), # parameter names which get converted to symbols
        (parameters = param_names, internals = internal_names); # name map (one needs to be called parameters = ...)
        start = discard_initial + 1,
        thin = thinning,
    )
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Vector{<:MCMCState}},
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.Ensemble,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial = 0,
    thinning = 1,
    param_names = missing,
    kwargs...,
)
    # Preallocate return array
    # NOTE: requires constant dimensionality.
    n_params = length(ts[1][1].params)
    nsamples = length(ts)
    # add 2 parameters for :log_density, :accepted
    vals = Array{Float64, 3}(undef, nsamples, n_params + 2, sampler.n_walkers)

    for n in 1:nsamples
        for i in 1:(sampler.n_walkers)
            walker = ts[n][i]
            for j in 1:n_params
                vals[n, j, i] = walker.params[j]
            end
            vals[n, n_params + 1, i] = walker.log_density
            vals[n, n_params + 2, i] = walker.accepted
        end
    end

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1][1].params))]
        #    elseif length(param_names) < length(keys(ts[1][1].params)) # in case bug with MV names, Chains still needs one name per dist.
        #        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1][1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end
    internal_names = [:log_density, :accepted]

    # Bundle everything up and return a MCChains.Chains struct.
    return MCMCChains.Chains(
        vals,
        vcat(param_names, internal_names),
        (parameters = param_names, internals = internal_names);
        start = discard_initial + 1,
        thin = thinning,
    )
end

# ------------------------------------------------------------------------------------------
# Top-level object to contain model and sampler (but not state)

"""
$(DocStringExtensions.TYPEDEF)

Top-level class holding all configuration information needed for MCMC sampling: the prior, 
emulated likelihood and sampling algorithm (`DensityModel` and `Sampler`, respectively, in 
AbstractMCMC's terminology).

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
struct MCMCWrapper{VV1 <: AbstractVector, VV2 <: AbstractVector, VV3 <: AbstractVector}
    "[`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) object describing the prior distribution on parameter values."
    prior::ParameterDistribution
    "[output_dim x N_samples] matrix, of given observation data."
    observations::VV1
    "Vector of observations describing the data samples to actually used during MCMC sampling (that have been transformed into a space consistent with emulator outputs)."
    encoded_observations::VV2
    "`AdvancedMH.DensityModel` object, used to evaluate the posterior density being sampled from."
    log_posterior_map::AbstractMCMC.AbstractModel
    "Object describing a MCMC sampling algorithm and its settings."
    mh_proposal_sampler::AbstractMCMC.AbstractSampler
    "NamedTuple of other arguments to be passed to `AbstractMCMC.sample()`."
    sample_kwargs::NamedTuple
    "Vector of encoders dictating how to encode/decode data."
    encoder_schedule::VV3
end

"""
$(TYPEDSIGNATURES)

gets the NameTuple of keywords that are passed into the Sampler algorithm
"""
get_sample_kwargs(mcmc::MCMCWrapper) = mcmc.sample_kwargs

"""
$(TYPEDSIGNATURES)

gets the stored `encoder_schedule` from an `MCMCWrapper`
"""
get_encoder_schedule(mcmc::MCMCWrapper) = mcmc.encoder_schedule


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructor for [`MCMCWrapper`](@ref) which performs the same standardization (SVD 
decorrelation) that was applied in the Emulator. It creates and wraps an instance of 
[`EmulatorPosteriorModel`](@ref), for sampling from the Emulator, and 
[`MetropolisHastingsSampler`](@ref), for generating the MC proposals.

- `mcmc_alg`: [`MCMCProtocol`](@ref) describing the MCMC sampling algorithm to use. Currently
  implemented algorithms are:

  - [`RWMHSampling`](@ref): Metropolis-Hastings sampling from a vanilla random walk with
    fixed stepsize.
  - [`pCNMHSampling`](@ref): Metropolis-Hastings sampling using the preconditioned 
    Crank-Nicholson algorithm, which has a well-behaved small-stepsize limit.
  - [`BarkerSampling`](@ref): Metropolis-Hastings sampling using the Barker
    proposal, which has a robustness to choosing step-size parameters.

- `obs_sample`: Vector (for one sample) or matrix with columns as samples from the observation. Can, e.g., be picked from an Observation struct using `get_obs_sample`.
- `prior`: [`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) 
  object containing the parameters' prior distributions.
- `emulator`: [`Emulator`](@ref) to sample from.
- `stepsize`: MCMC step size, applied as a scaling to the prior covariance.
- `init_params`: Starting parameter values for MCMC sampling.
- `burnin`: Initial number of MCMC steps to discard from output (pre-convergence).
"""
function MCMCWrapper(
    mcmc_alg::MCMCProtocol,
    observation::AMorAV,
    prior::ParameterDistribution,
    em_or_fmw::EorFMW;
    init_params::AV = [],
    burnin::Int = 0,
    kwargs...,
) where {
    AV <: AbstractVector,
    AMorAV <: Union{AbstractVector, AbstractMatrix},
    EorFMW <: Union{Emulator, ForwardMapWrapper},
}

    if length(init_params) == 0
        init_params = mean(prior)
    end

    # make into iterable over vectors
    obs_slice = if observation isa AbstractVector{<:AbstractVector}
        observation
    else # NB a vector is treated as a column here:
        eachcol(observation)
    end

    # encoding, saved in MCMCWrapper
    encoder_schedule = get_encoder_schedule(em_or_fmw)

    # encoding data works on columns but mcmc wants vec-of-vec
    encoded_obs = [vec(encode_data(encoder_schedule, reshape(obs, :, 1), "out")) for obs in obs_slice]
    # encoding initial condition
    encoded_init_params = vec(encode_data(encoder_schedule, reshape(init_params, :, 1), "in"))

    log_posterior_map = EmulatorPosteriorModel(prior, em_or_fmw, encoded_obs)
    mh_proposal_sampler = MetropolisHastingsSampler(mcmc_alg, prior)

    # naming encoded dimensions
    param_names = ["encoded_param_$(k)" for k in 1:size(encoded_init_params, 1)]

    sample_kwargs = (; # set defaults here
        :initial_params => deepcopy(encoded_init_params),
        :param_names => param_names,
        :discard_initial => burnin,
        :chain_type => MCMCChains.Chains,
    )
    sample_kwargs = merge(sample_kwargs, kwargs) # override defaults with any explicit values
    return MCMCWrapper(
        prior,
        obs_slice,
        encoded_obs,
        log_posterior_map,
        mh_proposal_sampler,
        sample_kwargs,
        encoder_schedule,
    )
end

function MCMCWrapper(
    mcmc_alg::MCMCProtocol,
    observation::OB,
    prior::PD,
    em_or_fmw::EorFMW;
    kwargs...,
) where {OB <: Observation, PD <: ParameterDistribution, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return MCMCWrapper(mcmc_alg, get_obs(observation), prior, em_or_fmw; kwargs...)
end

function MCMCWrapper(
    mcmc_alg::MCMCProtocol,
    observation_series::OS,
    prior::PD,
    em_or_fmw::EorFMW;
    kwargs...,
) where {OS <: ObservationSeries, PD <: ParameterDistribution, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    observations = [get_obs(ob) for ob in get_observations(observation_series)]
    return MCMCWrapper(mcmc_alg, observations, prior, em_or_fmw; kwargs...)
end
"""
   $(DocStringExtensions.FUNCTIONNAME)([rng,] mcmc::MCMCWrapper, args...; kwargs...)

Extends the `sample` methods of AbstractMCMC (which extends StatsBase) to sample from the 
emulated posterior, using the MCMC sampling algorithm and [`Emulator`](@ref) configured in 
[`MCMCWrapper`](@ref). Returns a [`MCMCChains.Chains`](https://beta.turing.ml/MCMCChains.jl/dev/) 
object containing the samples. 

Supported methods are:

- `sample([rng, ]mcmc, N; kwargs...)`

  Return a `MCMCChains.Chains` object containing `N` samples from the emulated posterior.

- `sample([rng, ]mcmc, isdone; kwargs...)`

  Sample from the `model` with the Markov chain Monte Carlo `sampler` until a convergence 
  criterion `isdone` returns `true`, and return the samples. The function `isdone` has the 
  signature

  ```julia
      isdone(rng, model, sampler, samples, state, iteration; kwargs...)
  ```

  where `state` and `iteration` are the current state and iteration of the sampler, 
  respectively. It should return `true` when sampling should end, and `false` otherwise.

- `sample([rng, ]mcmc, parallel_type, N, nchains; kwargs...)`

  Sample `nchains` Monte Carlo Markov chains in parallel according to `parallel_type`, which
  may be `MCMCThreads()` or `MCMCDistributed()` for thread and parallel sampling, 
  respectively.
"""
function sample(rng::Random.AbstractRNG, mcmc::MCMCWrapper, args...; kwargs...)
    # any explicit function kwargs override defaults in mcmc object
    kwargs = merge(mcmc.sample_kwargs, NamedTuple(kwargs))
    return AbstractMCMC.mcmcsample(rng, mcmc.log_posterior_map, mcmc.mh_proposal_sampler, args...; kwargs...)
end
# use default rng if none given
sample(mcmc::MCMCWrapper, args...; kwargs...) = sample(Random.GLOBAL_RNG, mcmc, args...; kwargs...)

# ------------------------------------------------------------------------------------------
# Search for a MCMC stepsize that yields a good MH acceptance rate

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fraction of MC proposals in `chain` which were accepted (according to Metropolis-Hastings.)
"""
function accept_ratio(chain::MCMCChains.Chains)
    if :accepted in names(chain, :internals)
        return mean(chain, :accepted)
    else
        throw("MH `accepted` not recorded in chain: $(names(chain, :internals)).")
    end
end

function _find_mcmc_step_log(mcmc::MCMCWrapper)
    str_ = @sprintf "%d starting params:" 0
    for p in zip(mcmc.sample_kwargs.param_names, mcmc.sample_kwargs.initial_params)
        str_ *= @sprintf " %s: %.3g" p[1] p[2]
    end
    println(str_)
    flush(stdout)
end

function _find_mcmc_step_log(it, stepsize, acc_ratio, chain::MCMCChains.Chains)
    str_ = @sprintf "%d stepsize: %.3g acc rate: %.3g\n\tparams:" it stepsize acc_ratio
    for p in pairs(get(chain; section = :parameters)) # can't map() over Pairs
        str_ *= @sprintf " %s: %.3g" p.first last(p.second)
    end
    println(str_)
    flush(stdout)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Uses a heuristic to return a stepsize for the `mh_proposal_sampler` element of 
[`MCMCWrapper`](@ref) which yields fast convergence of the Markov chain.

The criterion used is that Metropolis-Hastings proposals should be accepted between 15% and 
35% of the time.
"""
function optimize_stepsize(
    rng::Random.AbstractRNG,
    mcmc::MCMCWrapper;
    init_stepsize = 1.0,
    N = 2000,
    max_iter = 20,
    target_acc = 0.25,
    sample_kwargs...,
)
    increase = false
    decrease = false
    stepsize = init_stepsize
    factor = [1.0]
    step_history = [true, true]
    _find_mcmc_step_log(mcmc)
    for it in 1:max_iter
        trial_chain = sample(rng, mcmc, N; stepsize = stepsize, sample_kwargs...)
        acc_ratio = accept_ratio(trial_chain)
        _find_mcmc_step_log(it, stepsize, acc_ratio, trial_chain)

        change_step = true
        if acc_ratio < target_acc - 0.1
            decrease = true
        elseif acc_ratio > target_acc + 0.1
            increase = true
        else
            change_step = false
        end

        if increase && decrease
            factor[1] /= 2
            increase = false
            decrease = false
        end

        if acc_ratio < target_acc - 0.1
            stepsize *= 2^(-factor[1])
        elseif acc_ratio > target_acc + 0.1
            stepsize *= 2^(factor[1])
        end

        if change_step
            @printf "Set sampler to new stepsize: %.3g\n" stepsize
        else
            @printf "Returning optimized stepsize: %.3g\n" stepsize
            return stepsize
        end
    end
    throw("Failed to choose suitable stepsize in $(max_iter) iterations.")
end
# use default rng if none given
optimize_stepsize(mcmc::MCMCWrapper; kwargs...) = optimize_stepsize(Random.GLOBAL_RNG, mcmc; kwargs...)


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a `ParameterDistribution` object corresponding to the empirical distribution of the 
samples in `chain`.

!!! note
    This method does not currently support combining samples from multiple `Chains`.
"""
function get_posterior(mcmc::MCMCWrapper, chain::MCMCChains.Chains, boost_for_loss=0.001)
    p_names = get_name(mcmc.prior)
    p_slices = batch(mcmc.prior)
    flat_constraints = get_all_constraints(mcmc.prior)

    # Cast data in chain to a ParameterDistribution object. Data layout in Chain is an
    # (N_samples x n_params x n_chains) AxisArray, so samples are in rows.
    p_chain = Array(Chains(chain, :parameters)) # discard internal/diagnostic data

    # get the encoding schedule for decoding
    encoder_schedule = get_encoder_schedule(mcmc)
    
    # flatten samples from the chain for manipulation
    red_samples = reduce(vcat, [p_chain[:, slice, 1]' for slice in p_slices])
    
    # Perform a lift back to the full-space, that is aware of the correlation between reduced and null-space variables (through a Gaussian assumption)
    # Can be done by defining the deterministic Gaussin gain K, to decode, then if the null-space "(I-KE)" is big we also inject noise in the null space
    C = cov(mcmc.prior)
    m = reshape(mean(mcmc.prior), : ,1)
    E, b = get_encoder_from_schedule(encoder_schedule, "in") # encoder is affine: E*x + b
    E = Matrix(E)
    enc_m = (E*m + b)
    ECEt = cholesky(Symmetric(E * C * E' + 1e-12*I))
    K = C * E' * (ECEt \ I) # Gain
    # deterministic reconstruction with the Gain (should be similar to decode_data(...))
    full_samples = m .+ K*(red_samples .- enc_m)

    projection_loss = tr(C-K*E*C) / tr(C)
    if projection_loss > boost_for_loss # (default >0.1% of variance is lost in projecting)
        @info "As the variance lost in reduced space is sufficiently large: $(projection_loss) > boost_for_loss (= $(boost_for_loss)), we inject additional noise into the complement of the reduced space"
        # Conditionally sample from the null space to preserve correlations with the reduced samples
        Σ_cond = C - K * E * C
        L = cholesky(Symmetric(Σ_cond + 1e-12*I)).L
        null_samples = L * randn(size(C,1), size(red_samples,2))
        full_samples += null_samples
    end
    
    p_samples = [Samples(full_samples[slice,:], params_are_columns = true) for slice in p_slices]
     
    # live in same space as prior
    # checks if a function distribution, by looking at if the distribution is nested
    p_constraints = [
        !isa(get_distribution(mcmc.prior)[pn], ParameterDistribution) ? # if not func-dist
        flat_constraints[slice] : # constraints are slice
        get_all_constraints(get_distribution(mcmc.prior)[pn]) # get constraints of nested dist
        for (pn, slice) in zip(p_names, p_slices)
    ]

    # distributions created as atoms and pieced together
    posterior_distribution = combine_distributions([
        ParameterDistribution(ps, pc, pn) for (ps, pc, pn) in zip(p_samples, p_constraints, p_names)
    ])
    return posterior_distribution
end

# other diagnostic utilities
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the expected squared jump distance of the chain.
"""
function esjd(chain::MCMCChains.Chains)
    samples = chain.value[:, :, 1] # N_samples x N_params x n_chains
    n_samples, n_params = size(samples)
    esjd = zeros(Float64, n_params)
    for i in 2:n_samples
        esjd .+= (samples[i, :] .- samples[i - 1, :]) .^ 2 ./ n_samples
    end
    return esjd

end





end # module MarkovChainMonteCarlo
