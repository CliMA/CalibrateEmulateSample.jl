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
    em_or_fmw::EorFMW;
    init_params::AV,
    burnin::Int,
    kwargs...,
) where {
    AV <: AbstractVector,
    AMorAV <: Union{AbstractVector, AbstractMatrix},
    EorFMW <: Union{Emulator, ForwardMapWrapper},
}

sample
get_posterior
optimize_stepsize
get_sample_kwargs
get_encoder_schedule
```

See [AbstractMCMC sampling API](@ref) for background on our use of Turing.jl's 
[AbstractMCMC](https://turing.ml/dev/docs/for-developers/interface) API for 
MCMC sampling.

## Sampler algorithms

```@docs
MCMCProtocol
AutodiffProtocol
GradFreeProtocol
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

## Diagnostics

```@docs
esjd
```
