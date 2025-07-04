# MarkovChainMonteCarlo

```@meta
CurrentModule = CalibrateEmulateSample.MarkovChainMonteCarlo
```

## Top-level class and methods

```@docs
MCMCWrapper
MCMCWrapper(
    mcmc_alg::MCMCProtocol,
    observation::AMorAV,
    prior::ParameterDistribution,
    em::Emulator;
    kwargs...,
) where {AV <: AbstractVector, AMorAV <: Union{AbstractVector, AbstractMatrix}}

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
AutodiffProtocol
ForwardDiffProtocol
ReverseDiffProtocol
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
