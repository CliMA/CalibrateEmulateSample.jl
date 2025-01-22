# MarkovChainMonteCarlo

```@meta
CurrentModule = CalibrateEmulateSample.MarkovChainMonteCarlo
```

## Top-level class and methods

```@docs
MCMCWrapper
MCMCWrapper(mcmc_alg::MCMCProtocol, obs_sample::AbstractVector{FT}, prior::ParameterDistribution, em::Emulator;init_params::AbstractVector{FT}, burnin::IT, kwargs...) where {FT<:AbstractFloat, IT<:Integer}
sample
get_posterior
optimize_stepsize
```

See [AbstractMCMC sampling API](@ref) for background on our use of Turing.jl's 
[AbstractMCMC](https://turing.ml/dev/docs/for-developers/interface) API for 
MCMC sampling.

## Sampler algorithms

```@docs
MCMCProtocol
RWMHSampling
pCNMHSampling
BarkerSampling
MetropolisHastingsSampler
```

## Emulated posterior (Model)

```@docs
EmulatorPosteriorModel
```

## Internals - MCMC State

```@docs
MCMCState
accept_ratio
```

## Internals - Other

```@docs
to_decorrelated
```