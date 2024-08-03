# refactor-Sample
module MMCMC

using ..Emulators
using ..ParameterDistributions

import Distributions: sample # Reexport sample()
using Distributions
using DocStringExtensions
using LinearAlgebra
using Printf
using Random
using Statistics
using ForwardDiff

using MCMCChains
import AbstractMCMC: sample # Reexport sample()
using AbstractMCMC
import AdvancedMH

export EmulatorPosteriorModel,
    MetropolisHastingsSampler,
    MCMCProtocol,
    RWMHSampling,
    pCNMHSampling,
    MALASampling,
    BarkerSampling,
    MCMCWrapper,
    accept_ratio,
    optimize_stepsize,
    get_posterior,
    sample

# ------------------------------------------------------------------------------------------
# Output space transformations between original and SVD-decorrelated coordinates.
# Redundant with what's in Emulators.jl, but need to reimplement since we don't have
# access to obs_noise_cov

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Transform samples from the original (correlated) coordinate system to the SVD-decorrelated
coordinate system used by [`Emulator`](@ref). Used in the constructor for [`MCMCWrapper`](@ref).
"""
function to_decorrelated(data::AbstractMatrix{FT}, em::Emulator{FT}) where {FT <: AbstractFloat}
    if em.standardize_outputs && em.standardize_outputs_factors !== nothing
        # standardize() data by scale factors, if they were given
        data = data ./ em.standardize_outputs_factors
    end
    decomp = em.decomposition
    if decomp !== nothing
        # Use SVD decomposition of obs noise cov, if given, to transform data to
        # decorrelated coordinates.
        inv_sqrt_singvals = Diagonal(1.0 ./ sqrt.(decomp.S))
        return inv_sqrt_singvals * decomp.Vt * data
    else
        return data
    end
end
function to_decorrelated(data::AbstractVector{FT}, em::Emulator{FT}) where {FT <: AbstractFloat}
    # method for single sample
    out_data = to_decorrelated(reshape(data, :, 1), em)
    return vec(out_data)
end

# ------------------------------------------------------------------------------------------
# Sampler extensions to differentiate vanilla RW and pCN algorithms
#
# (Strictly speaking the difference between RW and pCN should be implemented at the level of
# the MH Sampler's Proposal, not by defining a new Sampler, since the former is where the
# only change is made. We do the latter here because doing the former would require more
# boilerplate code (repeating AdvancedMH/src/proposal.jl for the new Proposals)).

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
struct RWMHSampling <: MCMCProtocol end

struct RWMetropolisHastings{D} <: AdvancedMH.MHSampler
    proposal::D
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::RWMetropolisHastings,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)

function _get_proposal(prior::ParameterDistribution)
    # *only* use covariance of prior, not full distribution
    Σsqrt = sqrt(ParameterDistributions.cov(prior)) # rt_cov * MVN(0,I) avoids the posdef errors for MVN in Julia Distributions
    return AdvancedMH.RandomWalkProposal(Σsqrt * MvNormal(zeros(size(Σsqrt)[1]), I))
end

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
MetropolisHastingsSampler(::RWMHSampling, prior::ParameterDistribution) = RWMetropolisHastings(_get_proposal(prior))

"""
$(DocStringExtensions.TYPEDEF)

[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for
new parameters according to the preconditioned Crank-Nicholson (pCN) algorithm, which is
usable for MCMC in the *stepsize → 0* limit, unlike the vanilla random walk. Steps are based
on the covariance of `prior`.
"""
struct pCNMHSampling <: MCMCProtocol end

struct pCNMetropolisHastings{D} <: AdvancedMH.MHSampler
    proposal::D
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::pCNMetropolisHastings,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)

MetropolisHastingsSampler(::pCNMHSampling, prior::ParameterDistribution) = pCNMetropolisHastings(_get_proposal(prior))
"""
$(DocStringExtensions.TYPEDEF)

[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for
new parameters according to the MALA.
"""
struct MALASampling <: MCMCProtocol end

struct MetropolisAdjustedLangevin{D} <: AdvancedMH.MHSampler
    proposal::D
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::MetropolisAdjustedLangevin,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)

MetropolisHastingsSampler(::MALASampling, prior::ParameterDistribution) = MetropolisAdjustedLangevin(_get_proposal(prior))
"""
$(DocStringExtensions.TYPEDEF)

[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for
new parameters according to the Barker proposal.
"""
struct BarkerSampling <: MCMCProtocol end

struct BarkerMetropolisHastings{D} <: AdvancedMH.MHSampler
    proposal::D
end
# Define method needed by AdvancedMH for new Sampler
AdvancedMH.logratio_proposal_density(
    sampler::BarkerMetropolisHastings,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate,
) = AdvancedMH.logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)

MetropolisHastingsSampler(::BarkerSampling, prior::ParameterDistribution) = BarkerMetropolisHastings(_get_proposal(prior))

# ------------------------------------------------------------------------------------------
# Use emulated model in sampler

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Factory which constructs `AdvancedMH.DensityModel` objects given a prior on the model
parameters (`prior`) and an [`Emulator`](@ref) of the log-likelihood of the data given
parameters. Together these yield the log posterior density we're attempting to sample from
with the MCMC, which is the role of the `DensityModel` class in the `AbstractMCMC` interface.
"""
function EmulatorPosteriorModel(
    prior::ParameterDistribution,
    em::Emulator{FT},
    obs_sample::AbstractVector{FT},
) where {FT <: AbstractFloat}
return AdvancedMH.DensityModel(
        function (θ)
            # θ: model params we evaluate at; in original coords.
            # transform_to_real = false means g, g_cov, obs_sample are in decorrelated coords.
            #
            # Recall predict() written to return multiple N_samples: expects input to be a
            # Matrix with N_samples columns. Returned g is likewise a Matrix, and g_cov is a
            # Vector of N_samples covariance matrices. For MH, N_samples is always 1, so we
            # have to reshape()/re-cast input/output; simpler to do here than add a
            # predict() method.
            g, g_cov =
                Emulators.predict(em, reshape(θ, :, 1), transform_to_real = false, vector_rf_unstandardize = false)
            #TODO vector_rf will always unstandardize, but other methods will not, so we require this additional flag.

            if isa(g_cov[1], Real)
                return logpdf(MvNormal(obs_sample, g_cov[1] * I), vec(g)) + logpdf(prior, θ)
            else
                return logpdf(MvNormal(obs_sample, g_cov[1]), vec(g)) + logpdf(prior, θ)
            end

        end,
    )
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

# Calculate the log density of the model given some parameterization.
AdvancedMH.logdensity(model::AdvancedMH.DensityModel, t::MCMCState) = t.log_density

# AdvancedMH.transition() is only called to create a new proposal, so create a MCMCState
# with accepted = true since that object will only be used if proposal is accepted.
function AdvancedMH.transition(
    sampler::MHS,
    model::AdvancedMH.DensityModel,
    params,
    log_density::Real,
) where {MHS <: Union{BarkerMetropolisHastings, MetropolisAdjustedLangevin, pCNMetropolisHastings, RWMetropolisHastings}}
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

# method extending AdvancedMH.propose() for Metropolis-adjusted Langevin algorithm
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::MetropolisAdjustedLangevin,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
        # Compute the gradient of the log-density at the current state
        log_gradient = ForwardDiff.gradient(x -> AdvancedMH.logdensity(model, x), current_state.params)
        proposed_state = current_state.params .+ (stepsize / 2) .* log_gradient .+ sqrt(stepsize) * rand(rng, sampler.proposal)
        return proposed_state
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
    log_gradient = ForwardDiff.gradient(x -> AdvancedMH.logdensity(model, x), current_state.params)
    n = length(current_state.params)
    u = rand(rng, n)
    xi = rand(rng, sampler.proposal)
    b = u .< 1 ./ (1 + exp(- log_gradient .* xi))
    return current_state.params .+ b .* xi
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
    log_α =
        new_log_density - AdvancedMH.logdensity(model, current_state) +
        AdvancedMH.logratio_proposal_density(sampler, current_state, new_params)

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
struct MCMCWrapper
    "[`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) object describing the prior distribution on parameter values."
    prior::ParameterDistribution
    "`AdvancedMH.DensityModel` object, used to evaluate the posterior density being sampled from."
    log_posterior_map::AbstractMCMC.AbstractModel
    "Object describing a MCMC sampling algorithm and its settings."
    mh_proposal_sampler::AbstractMCMC.AbstractSampler
    "NamedTuple of other arguments to be passed to `AbstractMCMC.sample()`."
    sample_kwargs::NamedTuple
end

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
  - [`MALASampling`](@ref): Metropolis-Hastings sampling using the Metropolis
    -adjusted Langevin algorithm, which exploits the gradient information of the target.
  - [`BarkerSampling`](@ref): Metropolis-Hastings sampling using the preconditioned
    Crank-Nicholson algorithm, which has a robustness to choosing step-size parameters.

- `obs_sample`: A single sample from the observations. Can, e.g., be picked from an
  Observation struct using `get_obs_sample`.
- `prior`: [`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/)
  object containing the parameters' prior distributions.
- `em`: [`Emulator`](@ref) to sample from.
- `stepsize`: MCMC step size, applied as a scaling to the prior covariance.
- `init_params`: Starting parameter values for MCMC sampling.
- `burnin`: Initial number of MCMC steps to discard from output (pre-convergence).
"""
function MCMCWrapper(
    mcmc_alg::MCMCProtocol,
    obs_sample::AbstractVector{FT},
    prior::ParameterDistribution,
    em::Emulator;
    init_params::AbstractVector{FT},
    burnin::IT = 0,
    kwargs...,
) where {FT <: AbstractFloat, IT <: Integer}
    obs_sample = to_decorrelated(obs_sample, em)
    log_posterior_map = EmulatorPosteriorModel(prior, em, obs_sample)
    mh_proposal_sampler = MetropolisHastingsSampler(mcmc_alg, prior)

    # parameter names are needed in every dimension in a MCMCChains object needed for diagnostics
    # so create the duplicates here
    dd = get_dimensions(prior)
    if all(dd .== 1) # i.e if dd == [1, 1, 1, 1, 1], => all params are univariate
        param_names = get_name(prior)
    else # else use multiplicity to get still informative parameter names
        pn = get_name(prior)
        param_names = reduce(vcat, [(pn[k] * "_") .* map(x -> string(x), 1:dd[k]) for k in 1:length(pn)])
    end

    sample_kwargs = (; # set defaults here
        :init_params => deepcopy(init_params),
        :param_names => param_names,
        :discard_initial => burnin,
        :chain_type => MCMCChains.Chains,
    )
    sample_kwargs = merge(sample_kwargs, kwargs) # override defaults with any explicit values
    return MCMCWrapper(prior, log_posterior_map, mh_proposal_sampler, sample_kwargs)
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
    for p in zip(mcmc.sample_kwargs.param_names, mcmc.sample_kwargs.init_params)
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
        if acc_ratio < 0.15
            decrease = true
        elseif acc_ratio > 0.35
            increase = true
        else
            change_step = false
        end

        if increase && decrease
            factor[1] /= 2
            increase = false
            decrease = false
        end

        if acc_ratio < 0.15
            stepsize *= 2^(-factor[1])
        elseif acc_ratio > 0.35
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
function get_posterior(mcmc::MCMCWrapper, chain::MCMCChains.Chains)
    p_names = get_name(mcmc.prior)
    p_slices = batch(mcmc.prior)
    flat_constraints = get_all_constraints(mcmc.prior)
    # live in same space as prior
    p_constraints = [flat_constraints[slice] for slice in p_slices]

    # Cast data in chain to a ParameterDistribution object. Data layout in Chain is an
    # (N_samples x n_params x n_chains) AxisArray, so samples are in rows.
    p_chain = Array(Chains(chain, :parameters)) # discard internal/diagnostic data
    p_samples = [Samples(p_chain[:, slice, 1], params_are_columns = false) for slice in p_slices]

    # distributions created as atoms and pieced together
    posterior_distribution = combine_distributions([
        ParameterDistribution(ps, pc, pn) for (ps, pc, pn) in zip(p_samples, p_constraints, p_names)
    ])
    return posterior_distribution
end
end # module MMCMC
