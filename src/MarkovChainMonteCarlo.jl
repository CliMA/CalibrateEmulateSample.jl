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
$(TYPEDEF)

Type used to dispatch different autodifferentiation methods where different emulators have a different compatability with autodiff packages 
"""
abstract type AutodiffProtocol end

"""
$(TYPEDEF)

Type to construct samplers for emulators not compatible with autodifferentiation
"""
abstract type GradFreeProtocol <: AutodiffProtocol end

"""
$(TYPEDEF)

Type to construct samplers for emulators compatible with `ForwardDiff.jl` autodifferentiation
"""
abstract type ForwardDiffProtocol <: AutodiffProtocol end

"""
$(TYPEDEF)

Type to construct samplers for emulators compatible with `ReverseDiff.jl` autodifferentiation
"""
abstract type ReverseDiffProtocol <: AutodiffProtocol end
# ...to be implemented...
#=
abstract type ZygoteProtocol <: AutodiffProtocol end
abstract type EnzymeProtocol <: AutodiffProtocol end
=#

function _get_cholesky_factor(encoded_prior::ParameterDistribution, encoder_schedule::VV) where {VV <: AbstractVector}
    # We use the prior covariance to shape the proposal (in the encoded space),
    # as proposals are based on increments we do not need to shift the mean too.
    # The Cholesky factor L (C = LL') is the single source of truth every sampler below
    # uses both to draw its noise and to evaluate its transition (log-)density, so the
    # two can never drift apart.
    C = cov(encoded_prior)
    return cholesky(Symmetric(C)).L
end


"""
$(TYPEDEF)

Type used to dispatch different methods of the [`MetropolisHastingsSampler`](@ref) 
constructor, corresponding to different sampling algorithms.
"""
abstract type MCMCProtocol end

"""
$(TYPEDEF)
    
[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for 
new parameters as as vanilla random walk, based on the covariance of `prior`.
"""
struct RWMHSampling{T <: AutodiffProtocol} <: MCMCProtocol end

RWMHSampling() = RWMHSampling{GradFreeProtocol}()

struct RWMetropolisHastings{LT, ADT <: AutodiffProtocol} <: AdvancedMH.MHSampler
    "Lower Cholesky factor `L` of the (encoded) prior covariance `C = LL'`, shaping the proposal noise."
    cholesky_L::LT
end

"""
$(TYPEDSIGNATURES)

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
function MetropolisHastingsSampler(
    ::RWMHSampling{T},
    encoded_prior::ParameterDistribution,
    encoder_schedule::VV,
) where {T <: AutodiffProtocol, VV <: AbstractVector}
    L = _get_cholesky_factor(encoded_prior, encoder_schedule)
    return RWMetropolisHastings{typeof(L), T}(L)
end

"""
$(TYPEDEF)
    
[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for 
new parameters according to the preconditioned Crank-Nicholson (pCN) algorithm, which is 
usable for MCMC in the *stepsize → 0* limit, unlike the vanilla random walk. Steps are based 
on the covariance of `prior`.
"""
struct pCNMHSampling{T <: AutodiffProtocol} <: MCMCProtocol end
pCNMHSampling() = pCNMHSampling{GradFreeProtocol}()

struct pCNMetropolisHastings{LT, T <: AutodiffProtocol} <: AdvancedMH.MHSampler
    "Lower Cholesky factor `L` of the (encoded) prior covariance `C = LL'`, shaping the proposal noise."
    cholesky_L::LT
end
function MetropolisHastingsSampler(
    ::pCNMHSampling{T},
    encoded_prior::ParameterDistribution,
    encoder_schedule::VV,
) where {T <: AutodiffProtocol, VV <: AbstractVector}
    L = _get_cholesky_factor(encoded_prior, encoder_schedule)
    return pCNMetropolisHastings{typeof(L), T}(L)
end

#------ The following are gradient-based samplers

"""
$(TYPEDEF)

[`MCMCProtocol`](@ref) which uses Metropolis-Hastings sampling that generates proposals for
new parameters according to the Barker proposal.
"""
struct BarkerSampling{T <: AutodiffProtocol} <: MCMCProtocol end
BarkerSampling() = BarkerSampling{ForwardDiffProtocol}()

struct BarkerMetropolisHastings{LT, T <: AutodiffProtocol} <: AdvancedMH.MHSampler
    "Lower Cholesky factor `L` of the (encoded) prior covariance `C = LL'`, shaping the proposal noise."
    cholesky_L::LT
end

function MetropolisHastingsSampler(
    ::BarkerSampling{T},
    encoded_prior::ParameterDistribution,
    encoder_schedule::VV,
) where {T <: AutodiffProtocol, VV <: AbstractVector}
    L = _get_cholesky_factor(encoded_prior, encoder_schedule)
    return BarkerMetropolisHastings{typeof(L), T}(L)
end

# ------------------------------------------------------------------------------------------
# Shared Metropolis-Hastings transition-density interface.
#
# Every sampler above is required to implement a function,
# log_transition_density(sampler, model, θ_from, θ_to; stepsize) = log q(θ_to | θ_from)
# the forward transition log-density that sampler's `propose` draws from.
# there is no fallback

# AdvancedMH.logratio_proposal_density, fed into the MH acceptance ratio, is
# a shared logic among all AdvancedMH.MHSamplers. Applying the Hastings correction
# log q(θ_from | θ_to) − log q(θ_to | θ_from)  (reverse − forward), by
# evaluating log_transition_density in both directions. 
function log_transition_density(
    sampler::MHS,
    model,
    θ_from,
    θ_to;
    stepsize::FT = 1.0,
) where {MHS <: AdvancedMH.MHSampler, FT <: AbstractFloat}
    throw(
        ArgumentError(
            "log_transition_density not implemented for $(MHS). Every MHSampler in this module " *
            "must implement its own log_transition_density(sampler, model, θ_from, θ_to; stepsize) " *
            "= log q(θ_to | θ_from); there is no generic/symmetric fallback, since silently assuming " *
            "one caused the prior-double-counting bug this interface replaces.",
        ),
    )
end

function AdvancedMH.logratio_proposal_density(
    sampler::MHS,
    model::AdvancedMH.DensityModel,
    transition_prev::AdvancedMH.AbstractTransition,
    candidate;
    stepsize::FT = 1.0,
) where {MHS <: AdvancedMH.MHSampler, FT <: AbstractFloat}
    θ_from = transition_prev.params
    θ_to = candidate
    log_q_forward = log_transition_density(sampler, model, θ_from, θ_to; stepsize = stepsize)
    log_q_reverse = log_transition_density(sampler, model, θ_to, θ_from; stepsize = stepsize)
    return log_q_reverse - log_q_forward
end


## -------- Autodifferentiation procedures ------ ##

function autodiff_gradient(model::AdvancedMH.DensityModel, params, autodiff_protocol)
    if autodiff_protocol == ForwardDiffProtocol
        return ForwardDiff.gradient(x -> AdvancedMH.logdensity(model, x), params)
    elseif autodiff_protocol == ReverseDiffProtocol
        return ReverseDiff.gradient(x -> AdvancedMH.logdensity(model, x), params)
    else
        _throw_no_autodiff_impl(:gradient, autodiff_protocol)
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
        _throw_no_autodiff_impl(:hessian, autodiff_protocol)
    end
end

autodiff_hessian(model::AdvancedMH.DensityModel, params, sampler::MH) where {MH <: AdvancedMH.MHSampler} =
    autodiff_hessian(model, params, typeof(sampler).parameters[2])




# ------------------------------------------------------------------------------------------
# Use emulated model in sampler

"""
$(TYPEDSIGNATURES)

Defines the internal log-density function over a vector of observation samples using an assumed conditionally indepedent likelihood, that is with a log-likelihood of `ℓ(y,θ) = sum^n_i log( p(y_i|θ) )`.

Inputs:
=======
- θ: Parameters in unconstrained, and encoded coordinates.
- encoded_prior: Encoded prior distribution as a ParameterDistribution (defined on the unconstrained, and encoded coordinates)
- em_or_fmw: `Emulator` or `ForwardMapWrapper` object with predict(.) method
- obs_vec: encoded data vector sample(s)
"""
function emulator_log_density_model(
    θ,
    encoded_prior::ParameterDistribution,
    em_or_fmw::EorFMW,
    obs_vec::AV,
) where {AV <: AbstractVector, EorFMW <: Union{Emulator, ForwardMapWrapper}}

    # Returned g is a length-1, Vector{Real} or Vector{Vector}, and g_cov is length-1 Vector{Vector} or Vector{Matrix} respectively
    g, g_cov = Emulators.predict(em_or_fmw, reshape(θ, :, 1), encode = "in_and_out", add_obs_noise_cov = true)

    if isa(g_cov[1], Real)
        return sum([logpdf(MvNormal(obs, g_cov[1] * I), vec(g)) for obs in obs_vec]) + logpdf(encoded_prior, θ)
    else
        return sum([logpdf(MvNormal(obs, g_cov[1]), vec(g)) for obs in obs_vec]) + logpdf(encoded_prior, θ)
    end

end

"""
$(TYPEDSIGNATURES)

Factory which constructs `AdvancedMH.DensityModel` objects given an (Encoded) prior on the model 
parameters (`encoded_prior`) and an [`Emulator`](@ref) of the log-likelihood of the data given 
parameters. Together these yield the log posterior density we're attempting to sample from 
with the MCMC, which is the role of the `DensityModel` class in the `AbstractMCMC` interface.
"""
function EmulatorPosteriorModel(
    encoded_prior::ParameterDistribution,
    em_or_fmw::EorFMW,
    obs_vec::AV,
) where {AV <: AbstractVector, EorFMW <: Union{Emulator, ForwardMapWrapper}}

    return AdvancedMH.DensityModel(x -> emulator_log_density_model(x, encoded_prior, em_or_fmw, obs_vec))
end

# ------------------------------------------------------------------------------------------
# Record MH accept/reject decision in MCMCState object

"""
$(TYPEDEF)

Extends the `AdvancedMH.Transition` (which encodes the current state of the MC during
sampling) with a boolean flag to record whether this state is new (arising from accepting a
Metropolis-Hastings proposal) or old (from rejecting a proposal).

# Fields
$(TYPEDFIELDS)
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

# ------------------------------------------------------------------------------------------
# Markov transition kernels as Distributions.jl objects.
#
# For samplers whose transition kernel q(·|θ_from) is an ordinary Distributions.jl
# distribution, we go one step further than log_transition_density: rather than hand-writing
# `propose` (a sample) and `log_transition_density` (a logpdf) as two separate expressions that
# merely happen to describe the same kernel, we build the kernel ONCE as a Distributions.jl
# object and derive both `rand` and `logpdf` from it generically. The mean/covariance/ρ formulas
# then exist in exactly one place (`transition_kernel`) instead of two, so they cannot drift
# apart even in principle — this is strictly stronger than the log_transition_density interface
# above, which only guarantees forward/reverse agree with *each other*, not that `propose`
# agrees with either.
const DistributionKernelSampler = Union{RWMetropolisHastings, pCNMetropolisHastings}

function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::MHS,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0,
) where {MHS <: DistributionKernelSampler, FT <: AbstractFloat}
    return rand(rng, transition_kernel(sampler, model, current_state.params; stepsize = stepsize))
end

function log_transition_density(
    sampler::MHS,
    model::AdvancedMH.DensityModel,
    θ_from,
    θ_to;
    stepsize::FT = 1.0,
) where {MHS <: DistributionKernelSampler, FT <: AbstractFloat}
    return logpdf(transition_kernel(sampler, model, θ_from; stepsize = stepsize), θ_to)
end

# θ_to | θ_from ~ N(θ_from, stepsize² C) — the symmetric random-walk kernel.
function transition_kernel(
    sampler::RWMetropolisHastings,
    model::AdvancedMH.DensityModel,
    θ_from;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    L = sampler.cholesky_L
    return MvNormal(θ_from, Symmetric((stepsize^2) .* (L * L')))
end

# θ_to | θ_from ~ N(ρθ_from, (1-ρ²)C) — the asymmetric pCN kernel (Beskos et al. 2017 for the
# ρ-vs-Euler-stepsize relation). Evaluating this honestly in both directions (via the generic
# logratio_proposal_density) is what recovers the pCN cancellation log q(θ_from|θ_to) -
# log q(θ_to|θ_from) = logprior(θ_from) - logprior(θ_to), instead of the silently-symmetric 0
# the previous implementation returned.
function transition_kernel(
    sampler::pCNMetropolisHastings,
    model::AdvancedMH.DensityModel,
    θ_from;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    ρ = (1 - stepsize / 4) / (1 + stepsize / 4)
    L = sampler.cholesky_L
    return MvNormal(ρ .* θ_from, Symmetric((1 - ρ^2) .* (L * L')))
end

# The Barker transition kernel q(·|θ_from), as a lightweight (non-Distributions.jl) struct: the
# whitened-coordinate machinery (Livingstone & Zanella, 2022) needs a state- and model-dependent
# gradient computed once, so we bundle exactly what _barker_rand/_barker_logpdf need to stay
# consistent, and no more — unlike RW/pCN this isn't a Distributions.jl distribution (its density
# isn't a standard family), so we write the two functions directly instead of reimplementing
# Distributions.jl's dispatch machinery (_rand!/_logpdf/insupport/...) for a one-off type.
struct BarkerKernel{VT, MT, FT <: AbstractFloat}
    "Current point θ_from that the kernel proposes away from."
    θ_from::VT
    "Gradient of the target log-density at θ_from, rotated into whitened coordinates: L'∇log π(θ_from)."
    grad_white::VT
    "Cholesky factor L of the prior covariance (C = LL'), used to whiten/unwhiten."
    L::MT
    stepsize::FT
end

function transition_kernel(
    sampler::BarkerMetropolisHastings,
    model::AdvancedMH.DensityModel,
    θ_from;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    L = sampler.cholesky_L
    grad_white = L' * autodiff_gradient(model, θ_from, sampler)
    return BarkerKernel(θ_from, grad_white, L, stepsize)
end

# Livingstone and Zanella (2022). The elementwise sigmoid selection that defines the Barker
# proposal is only reversible for *independent* coordinates, so we apply it in the whitened
# coordinate system u = L⁻¹θ (where the prior-covariance preconditioning L is decorrelated to the
# identity) and map the increment back with L. We also flip the sign of the whitened noise η
# rather than zeroing it out — the standard Barker kernel moves by ±η, never by exactly 0, so its
# transition density (_barker_logpdf below) is an ordinary, atom-free density; zeroing out
# (keep-or-drop) instead creates a mixed discrete/continuous kernel with no closed form.
function _barker_rand(rng::Random.AbstractRNG, kernel::BarkerKernel)
    n = length(kernel.θ_from)
    η = randn(rng, n)
    flip_prob = 1 ./ (1 .+ exp.(-kernel.grad_white .* η))
    sign = 2 .* (rand(rng, n) .< flip_prob) .- 1
    ζ = kernel.stepsize .* sign .* η
    return kernel.θ_from .+ kernel.L * ζ
end

# For a symmetric noise density g (here standard normal) and whitened increment e = ζ/stepsize,
# the marginal density of the ± flip-sign move is q(e | θ_from) = 2 g(e) σ(grad_white·e) (Barker,
# 1965; Livingstone & Zanella, 2022): summing the probability that e itself was kept (w.p.
# σ(grad_white·e)) with the probability that -e was drawn and flipped (w.p. σ(grad_white·e) too,
# since g(-e) = g(e)). See _barker_rand above for how θ_to is generated from θ_from.
function _barker_logpdf(kernel::BarkerKernel, θ_to)
    e = (kernel.L \ (θ_to .- kernel.θ_from)) ./ kernel.stepsize
    return sum(log(2) .+ logpdf.(Normal(), e) .+ log.(1 ./ (1 .+ exp.(-kernel.grad_white .* e)))) -
           length(e) * log(kernel.stepsize)
end

# method extending AdvancedMH.propose() for the Barker proposal
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::BarkerMetropolisHastings,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    return _barker_rand(rng, transition_kernel(sampler, model, current_state.params; stepsize = stepsize))
end

function log_transition_density(
    sampler::BarkerMetropolisHastings,
    model::AdvancedMH.DensityModel,
    θ_from,
    θ_to;
    stepsize::FT = 1.0,
) where {FT <: AbstractFloat}
    return _barker_logpdf(transition_kernel(sampler, model, θ_from; stepsize = stepsize), θ_to)
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
        new_log_density - current_log_density +
        AdvancedMH.logratio_proposal_density(sampler, model, current_state, new_params; stepsize = stepsize)

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
$(TYPEDEF)

Top-level class holding all configuration information needed for MCMC sampling: the prior, 
emulated likelihood and sampling algorithm (`DensityModel` and `Sampler`, respectively, in 
AbstractMCMC's terminology).

# Fields
$(TYPEDFIELDS)
"""
struct MCMCWrapper{VV1 <: AbstractVector, VV2 <: AbstractVector, VV3 <: AbstractVector}
    "[`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) object describing the prior distribution on parameter values."
    prior::ParameterDistribution
    "[`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) object describing the encoded prior distribution on the encoded parameter values."
    encoded_prior::ParameterDistribution
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
$(TYPEDSIGNATURES)

Constructor for [`MCMCWrapper`](@ref) that will perform an MCMC sampling in the encoded space given by `em_or_fmw` (`Emulator` or `ForwardMapWrapper`). It creates and wraps an instance of 
[`EmulatorPosteriorModel`](@ref), for sampling from the Emulator (predicting means and covariances), and 
[`MetropolisHastingsSampler`](@ref), for generating the MC proposals.

- `mcmc_alg`: [`MCMCProtocol`](@ref) describing the MCMC sampling algorithm to use. Currently
  implemented algorithms are:

  - [`RWMHSampling`](@ref): Metropolis-Hastings sampling from a vanilla random walk with
    fixed stepsize.
  - [`pCNMHSampling`](@ref): Metropolis-Hastings sampling using the preconditioned 
    Crank-Nicholson algorithm, which has a well-behaved small-stepsize limit.
  - [`BarkerSampling`](@ref): Metropolis-Hastings sampling using the Barker
    proposal, which has a robustness to choosing step-size parameters.

- `observation`: Vector (for one sample) or matrix with columns as samples from the observation. Can, e.g., be picked from an `Observation` struct using `get_obs_sample`.
- `prior`: [`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) 
  object containing the parameters' prior distributions.
- `em_or_fmw`: [`Emulator`](@ref) or `ForwardMapWrapper` to sample from.
- `init_params`: Starting parameter values for MCMC sampling. (defined in unconstrained parameter coordinates)
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
        init_params = ndims(prior) > 1 ? mean(prior) : [mean(prior)]
    end

    # make into iterable over vectors
    obs_slice = if observation isa AbstractVector{<:AbstractVector}
        observation
    else # NB a vector is treated as a column here:
        eachcol(observation)
    end

    # encodings! Saved in MCMCWrapper
    # We encode (1) data, (2) initial params (3) prior
    encoder_schedule = get_encoder_schedule(em_or_fmw)

    # encoding data works on columns but mcmc wants vec-of-vec
    encoded_obs = [vec(encode_data(encoder_schedule, reshape(obs, :, 1), "out")) for obs in obs_slice]
    # encoding initial condition
    encoded_init_params = vec(encode_data(encoder_schedule, reshape(init_params, :, 1), "in"))
    # encoding the prior (Gaussian assumptions)
    mp = ndims(prior) == 1 ? [mean(prior)] : mean(prior)
    cp = cov(prior)
    encoded_mean = vec(encode_data(encoder_schedule, reshape(mp, :, 1), "in"))
    encoded_cov = Matrix(encode_structure_matrix(encoder_schedule, cov(prior), "in"))
    if size(encoded_init_params, 1) == 1 # 1D
        enc_dist = Parameterized(Normal(encoded_mean[1], sqrt(encoded_cov[1]))) #N(μ,σ)
    else
        enc_dist = Parameterized(MvNormal(encoded_mean, Symmetric(encoded_cov) + 1e-12I)) # N(m,0.5*(C+C'))
    end
    encoded_prior = ParameterDistribution(
        Dict(
            "distribution" => enc_dist,
            "constraint" => repeat([no_constraint()], length(encoded_mean)),
            "name" => "encoded_prior_gaussian_$(length(encoded_mean))D",
        ),
    )

    # pass in encoded prior here. (Only use decoded prior for final decoding of posterior)
    log_posterior_map = EmulatorPosteriorModel(encoded_prior, em_or_fmw, encoded_obs)
    mh_proposal_sampler = MetropolisHastingsSampler(mcmc_alg, encoded_prior, encoder_schedule)

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
        encoded_prior,
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
$(TYPEDSIGNATURES)

Extend the `AbstractMCMC.sample` interface to draw from the emulated posterior configured
in `mcmc`. Returns a `MCMCChains.Chains` object.

Supported call patterns:

- `sample([rng, ]mcmc, N; kwargs...)` — draw exactly `N` samples.
- `sample([rng, ]mcmc, isdone; kwargs...)` — sample until `isdone(rng, model, sampler, samples, state, iteration; kwargs...)` returns `true`.
- `sample([rng, ]mcmc, parallel_type, N, nchains; kwargs...)` — run `nchains` chains in parallel via `MCMCThreads()` or `MCMCDistributed()`.
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
$(TYPEDSIGNATURES)

Fraction of MC proposals in `chain` which were accepted (according to Metropolis-Hastings.)
"""
function accept_ratio(chain::MCMCChains.Chains)
    if :accepted in names(chain, :internals)
        return mean(chain, :accepted)
    else
        error("MH `:accepted` not recorded in MCMC chain — available internals: $(names(chain, :internals)).")
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
$(TYPEDSIGNATURES)

Uses a bracket-and-bisect search to return a stepsize for the `mh_proposal_sampler` element of
[`MCMCWrapper`](@ref) which yields fast convergence of the Markov chain.

The criterion used is that Metropolis-Hastings proposals should be accepted between
`target_acc - tol` and `target_acc + tol` of the time (`tol = 0.1` by default). Acceptance rate
is assumed to (eventually) decrease as `stepsize` increases, but is not assumed to do so at any
particular rate: some proposals (e.g. gradient-informed ones such as the Barker proposal) can
have a much steeper acceptance-vs-stepsize transition than a vanilla random walk. The search
therefore first exponentially expands a bracket `[lo, hi]` around `init_stepsize` until `target_acc`
is known to lie strictly between the two trial acceptance rates, then bisects (in log-stepsize
space, i.e. by geometric mean) within that bracket. Bisection halves the bracket every iteration
regardless of how steep the transition is, so this converges reliably in cases where a fixed-factor
doubling/halving search can overshoot the target window and fail to converge within `max_iter`.

The exponential expansion phase is capped separately at `max_expansions` doublings/halvings
(default 10, i.e. a ~1000x range around `init_stepsize`). Acceptance rate computed from a noisy,
numerically-evaluated log-density (e.g. from an emulator) need not actually decrease monotonically
all the way to stepsize → 0 — very small proposals can be dominated by the log-density's own
floating-point noise floor rather than its true (smooth) small-stepsize limit, degrading rather
than improving acceptance. If expansion hasn't bracketed `target_acc` within `max_expansions`
steps, that is reported directly (`target_acc` may not be achievable in this range) rather than
continuing to shrink/grow the stepsize toward a pathological value.
"""
function optimize_stepsize(
    rng::Random.AbstractRNG,
    mcmc::MCMCWrapper;
    init_stepsize = 1.0,
    N = 2000,
    max_iter = 20,
    max_expansions = 10,
    target_acc = 0.25,
    tol = 0.1,
    sample_kwargs...,
)
    _find_mcmc_step_log(mcmc)

    n_evals = 0
    function acc_at(stepsize)
        n_evals += 1
        if n_evals > max_iter
            error(
                "optimize_stepsize: acceptance rate did not reach target $(target_acc) ± $(tol) " *
                "within $(max_iter) iterations — last stepsize tried: $(round(stepsize; sigdigits = 3)).",
            )
        end
        trial_chain = sample(rng, mcmc, N; stepsize = stepsize, sample_kwargs...)
        acc_ratio = accept_ratio(trial_chain)
        _find_mcmc_step_log(n_evals, stepsize, acc_ratio, trial_chain)
        return acc_ratio
    end

    function done(stepsize)
        @printf "Returning optimized stepsize: %.3g\n" stepsize
        return stepsize
    end
    within_tol(acc) = abs(acc - target_acc) <= tol

    stepsize = init_stepsize
    acc = acc_at(stepsize)
    within_tol(acc) && return done(stepsize)

    # Phase 1: exponentially expand a bracket [lo, hi] around init_stepsize, keeping the tightest
    # known too-high-acceptance point (lo) and too-low-acceptance point (hi), until both are known.
    # Capped at max_expansions: if acceptance hasn't crossed target_acc by then, it likely isn't
    # monotonic in this range (rather than simply needing one more doubling/halving), so we stop
    # and report that clearly instead of continuing toward a pathological stepsize.
    if acc > target_acc
        lo, hi = stepsize, stepsize
        acc_hi = acc
        n_expand = 0
        while acc_hi > target_acc
            n_expand += 1
            if n_expand > max_expansions
                error(
                    "optimize_stepsize: acceptance rate stayed above target $(target_acc) ± $(tol) " *
                    "after $(max_expansions) doublings (up to stepsize $(round(hi; sigdigits = 3))) " *
                    "without crossing it — the acceptance-vs-stepsize relationship may not be monotonic " *
                    "in this range rather than simply needing a larger stepsize.",
                )
            end
            lo = hi
            hi *= 2
            acc_hi = acc_at(hi)
            within_tol(acc_hi) && return done(hi)
        end
    else
        lo, hi = stepsize, stepsize
        acc_lo = acc
        n_expand = 0
        while acc_lo < target_acc
            n_expand += 1
            if n_expand > max_expansions
                error(
                    "optimize_stepsize: acceptance rate stayed below target $(target_acc) ± $(tol) " *
                    "after $(max_expansions) halvings (down to stepsize $(round(lo; sigdigits = 3))) " *
                    "without crossing it — the acceptance-vs-stepsize relationship may not be monotonic " *
                    "in this range (e.g. a numerically noisy log-density at very small stepsizes) rather " *
                    "than simply needing a smaller stepsize.",
                )
            end
            hi = lo
            lo /= 2
            acc_lo = acc_at(lo)
            within_tol(acc_lo) && return done(lo)
        end
    end

    # Phase 2: bisect within [lo, hi] (geometric mean, since stepsize tuning is multiplicative).
    while true
        mid = sqrt(lo * hi)
        acc_mid = acc_at(mid)
        within_tol(acc_mid) && return done(mid)
        if acc_mid > target_acc
            lo = mid
        else
            hi = mid
        end
    end
end
# use default rng if none given
optimize_stepsize(mcmc::MCMCWrapper; kwargs...) = optimize_stepsize(Random.GLOBAL_RNG, mcmc; kwargs...)

"""
$(TYPEDSIGNATURES)

Returns a `ParameterDistribution` object corresponding to the empirical distribution of the 
samples in `chain`.

keyword args
- `noise_injector_threshold`[`=0.001`]: If the encoded space is lossy, and the lost variability due to encoding exceeds a threshold `noise_injector_threshold`, then in place of decoding posterior samples, additional noise consistent with the prior is injected into the null space of the encoder. See `decode_and_add_noise()` for more detail.  
- `noise_injector_scaling`[`=1.0`]: Scales the injected noise; though 1.0 is the only "consistent" value, reduction may be necessary if noise injection causes posterior samples to be unstable in simulations.

!!! note
    This method does not currently support combining samples from multiple `Chains`.
"""
function get_posterior(
    mcmc::MCMCWrapper,
    chain::MCMCChains.Chains;
    noise_injector_threshold::FT = 0.001,
    noise_injector_scaling::FT = 1.0,
) where {FT <: Real}
    p_names = get_name(mcmc.prior)
    p_slices = batch(mcmc.prior)
    flat_constraints = get_all_constraints(mcmc.prior)

    # Cast data in chain to a ParameterDistribution object. Data layout in Chain is an
    # (N_samples x n_params x n_chains) AxisArray, so samples are in rows.
    p_chain = Array(Chains(chain, :parameters)) # discard internal/diagnostic data

    # get the encoding schedule for decoding
    encoder_schedule = get_encoder_schedule(mcmc)

    # flatten samples from the chain for manipulation as an array with columns as samples
    red_samples = p_chain[:, :, 1]'
    full_samples = decode_and_add_noise(
        encoder_schedule,
        red_samples,
        mcmc.prior,
        noise_injector_threshold,
        noise_injector_scaling,
    )

    p_samples = [Samples(full_samples[slice, :], params_are_columns = true) for slice in p_slices]

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
$(TYPEDSIGNATURES)

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

## Error helpers

@noinline function _throw_no_autodiff_impl(op::Symbol, autodiff_protocol)
    throw(ArgumentError("""
`autodiff_$(op)` called on a sampler with no $(op) implementation for protocol $(autodiff_protocol).

Got:
    autodiff_protocol = $(autodiff_protocol)

Suggestion:
    Select a protocol with a $(op) implementation, e.g. `ForwardDiffProtocol`.
"""))
end




end # module MarkovChainMonteCarlo
