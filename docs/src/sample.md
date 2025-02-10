# The Sample stage

```@meta
CurrentModule = CalibrateEmulateSample.MarkovChainMonteCarlo
```

The "sample" part of CES refers to exact sampling from the emulated posterior, in our current framework this is achieved with a [Markov chain Monte
Carlo algorithm](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC). Within this paradigm, we want to provide the flexibility to use multiple sampling algorithms; the approach we take is to use the general-purpose [AbstractMCMC.jl](https://turing.ml/dev/docs/for-developers/interface) API, provided by the [Turing.jl](https://turing.ml/dev/) probabilistic programming framework.


## User interface

We briefly outline an instance of how one sets up and uses MCMC within the CES package. The user first loads the MCMC module, and provides one of the Protocols (i.e. how one wishes to generate sampling proposals)

```julia
using CalibrateEmulateSample.MarkovChainMonteCarlo
protocol = RWMHSampling() # Random-Walk algorithm
# protocol = pCNMHSampling() # preconditioned-Crank-Nicholson algorithm
```
Then one builds the MCMC by providing the standard Bayesian ingredients (prior and data) from the calibrate stage, alongside the trained statistical emulator from the emulate stage:
```julia
mcmc = MCMCWrapper(
    protocol,
    truth_sample, 
    prior,
    emulator;
    init_params=mean_u_final,
    burnin=10_000,
)
```
The keyword arguments `init_params` give a starting step of the chain (often taken to be the mean of the final iteration of calibrate stage), and a `burnin` gives a number of initial steps to be discarded when drawing statistics from the sampling method.

!!! note "for many samples"
    If one has several samples of conditionally-independent data (that is, ``p({y_1,\dots,y_n}\mid\theta)`` is a product of ``\prod_i p(y_i\mid\theta)``), then one can feed in `truth_sample` as a vector of these samples ``y_ii`` or as a matrix. The resulting sampler will evaluate the likelihood at all y_i for every sample step. 

For good efficiency, one often needs to run MCMC with a problem-dependent step size. We provide a simple utility to help choose this. Here the optimizer runs short chains (of length `N`), and adjusts the step-size until the MCMC acceptance rate falls within an acceptable range, returning this step size.
```julia
new_step = optimize_stepsize(
mcmc;
init_stepsize = 1,
N = 2000
)
```
To generate ``10^5`` samples with a given step size (and optional random number generator `rng`), one calls
```julia
chain = sample(rng, mcmc, 100_000; stepsize = new_step)
display(chain) # gives diagnostics
```
The return argument is stored in an `MCMCChains.Chains` object. To convert this back into a `ParameterDistribution` type (which contains e.g. the transformation maps) one can call
```julia
posterior = get_posterior(mcmc, chain)
constrained_posterior = transform_unconstrained_to_constrained(prior, get_distribution(posterior))
```

One can quickly plot the marginals of the prior and posterior distribution with
```julia
using Plots
plot(prior)
plot!(posterior)
```
or extract statistics of the (unconstrained) distribution with
```julia
mean_posterior = mean(posterior)
cov_posterior = cov(posterior)
```

# [Further details on the implementation](@id AbstractMCMC sampling API)

This page provides a summary of AbstractMCMC which augments the existing documentation
(\[[1](https://turing.ml/dev/docs/for-developers/interface)\],
\[[2](https://turing.ml/dev/docs/for-developers/how_turing_implements_abstractmcmc)\]) and highlights how it's used by
the CES package in [MarkovChainMonteCarlo](@ref). It's not meant to be a complete description of the AbstractMCMC
package.

## Use in MarkovChainMonteCarlo

At present, Turing has limited support for derivative-free optimization, so we only use this abstract API and not Turing
itself. We also use two related dependencies, [AdvancedMH](https://github.com/TuringLang/AdvancedMH.jl) and
[MCMCChains](https://github.com/TuringLang/MCMCChains.jl). 

Julia's philosophy is to use small, composable packages rather than monoliths, but this can make it difficult to
remember where methods are defined! Below we describe the relevant parts of 

- The AbstractMCMC API,
- Extended to the case of Metropolis-Hastings (MH) sampling by AdvancedMH,
- Further extended for the needs of CES in [Markov chain Monte
  Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo).

### Sampler

A Sampler is AbstractMCMC's term for an implementation of a MCMC sampling algorithm, along with all its configuration
parameters. All samplers are a subtype of AbstractMCMC's `AbstractSampler`. 

Currently CES only implements the Metropolis-Hastings (MH) algorithm. Because it's so straightforward, much of
AbstractMCMC isn't needed. We implement two variants of MH with two different Samplers: `RWMetropolisHastings` and
`pCNMetropolisHastings`, both of which are subtypes of `AdvancedMH.MHSampler`. The constructor for
both Samplers is [`MetropolisHastingsSampler`](@ref); the different Samplers are specified by passing a
[`MCMCProtocol`](@ref) object to this constructor.

The `MHSampler` has only one field, `proposal`, the distribution used to generate new MH proposals via
additive stochastic perturbations to the current parameter values. This is done by
[AdvancedMH.propose()](https://github.com/TuringLang/AdvancedMH.jl/blob/master/src/proposal.jl), which gets called for
each MCMC `step()`. The difference between Samplers comes from how the proposal is generated:

- [`RWMHSampling`](@ref) does vanilla random-walk proposal generation with a constant, user-specified step size (this
  differs from the AdvancedMH implementation, which doesn't provide for a step size.)

- [`pCNMHSampling`](@ref) for preconditioned Crank-Nicholson proposals. Vanilla random walk sampling doesn't have a
  well-defined limit for high-dimensional parameter spaces; pCN replaces the random walk with an Ornstein–Uhlenbeck
  [AR(1)] process so that the Metropolis acceptance probability remains non-zero in this limit. See [Beskos et. al.
  (2008)](https://www.worldscientific.com/doi/abs/10.1142/S0219493708002378) and [Cotter et. al.
  (2013)](https://projecteuclid.org/journals/statistical-science/volume-28/issue-3/MCMC-Methods-for-Functions--Modifying-Old-Algorithms-to-Make/10.1214/13-STS421.full).

Generated proposals are then either accepted or rejected according to the same MH criterion
(in `step()`, below.)

### Models

In Turing, the Model is the distribution one performs inference on, which may involve observed and hidden variables and
parameters. For CES, we simply want to sample from the posterior, so our Model distribution is simply the emulated
likelihood (see [Emulators](@ref)) together with the prior. This is constructed by [`EmulatorPosteriorModel`](@ref).

### Sampling with the MCMC Wrapper object

At a [high level](https://turing.ml/dev/docs/using-turing/guide), a Sampler and Model is all that's needed to do MCMC
sampling. This is done by the [`sample`](https://turinglang.org/AbstractMCMC.jl/dev/api/#Sampling-a-single-chain) method
provided by AbstractMCMC (extending the method from BaseStats). 

To be more user-friendly, in CES we wrap the Sampler, Model and other necessary configuration into a
[`MCMCWrapper`](@ref) object. The constructor for this object ensures that all its components are created consistently,
and performs necessary bookkeeping, such as converting coordinates to the decorrelated basis. We extend [`sample`](@ref)
with methods to use this object (that simply unpack its fields and call the appropriate method from AbstractMCMC.)

### Chain

The [MCMCChain](https://beta.turing.ml/MCMCChains.jl/dev/) package provides the `Chains` container to store the results of the MCMC sampling; the package provides methods to for quick diagnostics and plot utilities of the the `Chains` objects. For example,

```julia
using MCMCChains
using StatsPlots

# ... from our MCMC example above ...
# chain = sample(rng, mcmc, 100_000; stepsize = new_step)

display(chain) # diagnostics
plot(chain) # plots samples over iteration and PDFs for each parameter
```


### Internals: Transitions

Implementing MCMC involves defining states and transitions of a Markov process (whose stationary distribution is what we
seek to sample from). AbstractMCMC's terminology is a bit confusing for the MH case; *states* of the chain are described
by `Transition` objects, which contain the current sample (and other information like its log-probability). 

AdvancedMH defines an `AbstractTransition` base class for use with its methods; we implement our own child class,
[`MCMCState`](@ref), in order to record statistics on the MH acceptance ratio.

### Internals: Markov steps

Markov *transitions* of the chain are defined by overloading AbstractMCMC's `step` method, which takes the Sampler and
current `Transition` and implements the Sampler's logic to returns an updated `Transition` representing the chain's new
state (actually, a pair of `Transitions`, for cases where the Sampler doesn't obey detailed balance; this isn't relevant
for us). 

For example, in Metropolis-Hastings sampling this is where we draw a proposal sample and accept or reject it according
to the MH criterion. AdvancedMH implements this
[here](https://github.com/TuringLang/AdvancedMH.jl/blob/ba86e49a3ebd1ee94d0becc3211738e3be6fd538/src/mh-core.jl#L90-L113);
we re-implement this method because 1) we need to record whether a proposal was accepted or rejected, and 2) our calls
to `propose()` are stepsize-dependent.

