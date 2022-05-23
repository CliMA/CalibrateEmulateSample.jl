# CalibrateEmulateSample.jl

`CalibrateEmulateSample.jl` solves parameter estimation problems using accelerated (and approximate) Bayesian inversion.

The framework can be applied currently to learn:
- the joint distribution for a moderate numbers of parameters (<40),
- it is not inherently restricted to unimodal distributions.

it can be used with computer models that:
- can be noisy or chaotic,
- are non-differentiable,
- can only be treated as black-box (interfaced only with parameter files).

The computer model is supplied by the user, as a parameter-to-data map ``G(u): \mathbb{R}^p \rightarrow \mathbb{R}^d``. For example, ``G`` could be a map from any given parameter configuration ``u`` to a collection of statistics of a dynamical system trajectory.

The data produced by the forward model are compared to observations $y$, which are assumed to be corrupted by additive noise ``\eta``, such that
```math
y = G(u) + \eta,
```
where the noise ``\eta`` is drawn from a d-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``.

### The inverse problem

Given an observation ``y``, the computer model ``G``, the observational noise ``\Gamma_y``, and some broad prior information on ``u``, we return the joint distribution of a data-informed distribution for "``u`` given ``y``".
 
As the name suggests, `CalibrateEmulateSample.jl` breaks this problem into a sequence of three steps: calibration, emulation, and sampling. A comprehensive treatment of the calibrate-emulate-sample approach to Bayesian inverse problems can be found in [Cleary et al., 2020](https://arxiv.org/pdf/2001.03689.pdf).

### The three steps of the algorithm:

The **calibrate** step of the algorithm consists of an application of [Ensemble Kalman Processes](https://github.com/CliMA/EnsembleKalmanProcesses.jl), that generates input-output pairs in high density around an optimal parameter ``u^*``. This ``u^*`` will be near a mode of the posterior distribution (Note: This the only time we interface with the forward model ``G``).

The **emulate** step takes these pairs and trains a statistical surrogate model (Gaussian process), emulating the forward map ``G``

The **sample** step uses this surrogate in place of ``G`` in a sampling method (Markov chain Monte Carlo) to sample the posterior distribution of ``u``.

`CalibrateEmulateSample.jl` contains the following modules:

Module                       | Purpose
-----------------------------|--------------------------------------------------------
CalibrateEmulateSample.jl    | Pulls in the [Ensemble Kalman Processes](https://github.com/CliMA/EnsembleKalmanProcesses.jl) package
Emulator.jl                  | Emulate: Modular template for emulators
GaussianProcess.jl           | - A Gaussian process emulator
MarkovChainMonteCarlo.jl     | Sample: Modular template for MCMC 
Utilities.jl                 | Helper functions

**The best way to get started is to have a look at the examples!**

## Authors

`CalibrateEmulateSample.jl` is being developed by the [Climate Modeling
Alliance](https://clima.caltech.edu).