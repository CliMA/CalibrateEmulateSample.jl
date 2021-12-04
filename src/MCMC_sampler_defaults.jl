# Default methods (and other boilerplate) for the AbstractMCMC interface.
#
# "Default" here means the logic in this file is independent of the specific samplers
# implemented in MCMC_samplers.jl. 
# Code here is taken directly from AdvancedMH.jl; we don't simply add that package as a 
# dependency because the fraction of code we need is relatively small. 


# ------------------------------------------------------------------------------------------
# Associated methods for Proposal objects

# Random draws
Base.rand(p::Proposal, args...) = rand(Random.GLOBAL_RNG, p, args...)
Base.rand(rng::Random.AbstractRNG, p::Proposal{<:Distribution}) = rand(rng, p.proposal)
function Base.rand(rng::Random.AbstractRNG, p::Proposal{<:AbstractArray})
    return map(x -> rand(rng, x), p.proposal)
end

# Densities
Distributions.logpdf(p::Proposal{<:Distribution}, v) = logpdf(p.proposal, v)
function Distributions.logpdf(p::Proposal{<:AbstractArray}, v)
    # `mapreduce` with multiple iterators requires Julia 1.2 or later
    return mapreduce(((pi, vi),) -> logpdf(pi, vi), +, zip(p.proposal, v))
end

# Proposals
function propose(rng::Random.AbstractRNG, proposal::Proposal{<:Function}, model::DensityModel)
    return propose(rng, proposal(), model)
end

function propose(rng::Random.AbstractRNG, proposal::Proposal{<:Function}, model::DensityModel, t)
    return propose(rng, proposal(t), model)
end

function q(proposal::Proposal{<:Function}, t, t_cond)
    return q(proposal(t_cond), t, t_cond)
end

# ------------------------------------------------------------------------------------------
# Extension of propose() method to multiple proposals

function propose(rng::Random.AbstractRNG, proposals::AbstractArray{<:Proposal}, model::DensityModel)
    return map(proposals) do proposal
        return propose(rng, proposal, model)
    end
end
function propose(rng::Random.AbstractRNG, proposals::AbstractArray{<:Proposal}, model::DensityModel, ts)
    return map(proposals, ts) do proposal, t
        return propose(rng, proposal, model, t)
    end
end

@generated function propose(rng::Random.AbstractRNG, proposals::NamedTuple{names}, model::DensityModel) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[:($name = propose(rng, proposals.$name, model)) for name in names]
    return expr
end

@generated function propose(
    rng::Random.AbstractRNG,
    proposals::NamedTuple{names},
    model::DensityModel,
    ts,
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[:($name = propose(rng, proposals.$name, model, ts.$name)) for name in names]
    return expr
end

# ------------------------------------------------------------------------------------------
# logratio_proposal_density methods
#
# This is logic for the fully general case (non-symmetric, i.e. breaking detailed balance).
# For the Proposals currently implemented, these methods will be overridden by ones that
# simply return 0.

function logratio_proposal_density(proposal::Proposal, state, candidate)
    return q(proposal, state, candidate) - q(proposal, candidate, state)
end

function logratio_proposal_density(sampler::MetropolisHastings, transition_prev::AbstractTransition, candidate)
    return logratio_proposal_density(sampler.proposal, transition_prev.params, candidate)
end

# type stable implementation for `NamedTuple`s
function logratio_proposal_density(
    proposals::NamedTuple{names},
    states::NamedTuple,
    candidates::NamedTuple,
) where {names}
    if @generated
        args = map(names) do name
            :(logratio_proposal_density(
                proposals[$(QuoteNode(name))],
                states[$(QuoteNode(name))],
                candidates[$(QuoteNode(name))],
            ))
        end
        return :(+($(args...)))
    else
        return sum(names) do name
            return logratio_proposal_density(proposals[name], states[name], candidates[name])
        end
    end
end

# use recursion for `Tuple`s to ensure type stability
logratio_proposal_density(proposals::Tuple{}, states::Tuple, candidates::Tuple) = 0

function logratio_proposal_density(proposals::Tuple{<:Proposal}, states::Tuple, candidates::Tuple)
    return logratio_proposal_density(first(proposals), first(states), first(candidates))
end

function logratio_proposal_density(proposals::Tuple, states::Tuple, candidates::Tuple)
    valfirst = logratio_proposal_density(first(proposals), first(states), first(candidates))
    valtail = logratio_proposal_density(Base.tail(proposals), Base.tail(states), Base.tail(candidates))
    return valfirst + valtail
end

# fallback for general iterators (arrays etc.) - possibly not type stable!
function logratio_proposal_density(proposals, states, candidates)
    return sum(zip(proposals, states, candidates)) do (proposal, state, candidate)
        return logratio_proposal_density(proposal, state, candidate)
    end
end

# ------------------------------------------------------------------------------------------
# bundle_samples() methods 
#
# Used to capture sampler history, using MCMCChains. 
# See AdvancedMH.jl/src/mcmcchains-connect.jl

"""
A basic MCMC.Chains constructor that works with the `AbstractTransition` struct defined
in AdvancedMH.jl.
"""
function AbstractMCMC.bundle_samples(
    ts::Vector{<:AbstractTransition},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names = missing,
    kwargs...,
)
    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["param_$i" for i in 1:length(keys(ts[1].params))]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    push!(param_names, "lp")

    # Turn all the transitions into a vector-of-NamedTuple.
    ks = tuple(Symbol.(param_names)...)
    nts = [NamedTuple{ks}(tuple(t.params..., t.lp)) for t in ts]
    return nts
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:AbstractTransition},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial = 0,
    thinning = 1,
    param_names = missing,
    kwargs...,
)
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.params, t.lp) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Bundle everything up and return a Chains struct.
    return MCMCChains.Chains(
        vals,
        vcat(param_names, [:lp]),
        (parameters = param_names, internals = [:lp]);
        start = discard_initial + 1,
        thin = thinning,
    )
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition{<:NamedTuple}},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial = 0,
    thinning = 1,
    param_names = missing,
    kwargs...,
)
    # Convert to a Vector{NamedTuple} first
    nts =
        AbstractMCMC.bundle_samples(ts, model, sampler, state, Vector{NamedTuple}; param_names = param_names, kwargs...)

    # Get all the keys
    all_keys = unique(mapreduce(collect ∘ keys, vcat, nts))

    # Push linearized draws onto array
    trygetproperty(thing, key) = key in keys(thing) ? getproperty(thing, key) : missing
    vals = map(nt -> [trygetproperty(nt, k) for k in all_keys], nts)

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = all_keys
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    # Bundle everything up and return a Chains struct.
    return MCMCChains.Chains(
        vals,
        param_names,
        (parameters = param_names, internals = [:lp]);
        start = discard_initial + 1,
        thin = thinning,
    )
end

"""
If the element type of `ts`` is `NamedTuple`s, just use the names in the struct.
"""
function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition{<:NamedTuple}},
    model::DensityModel,
    sampler::MHSampler,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names = missing,
    kwargs...,
)
    # Extract NamedTuples
    nts = map(x -> merge(x.params, (lp = x.lp,)), ts)
    return nts
end
