# CalibrateEmulateSample.jl

`CalibrateEmulateSample.jl` solves parameter estimation problems using accelerated (approximate) Bayesian inversion. The framework can be applied currently to learn the joint distribution for a moderate numbers of parameters (<40), and for computer models which can be noisy, chaotic, and are non-differentiable. For optimization only, please see our companion code [EnsembleKalmanProcesses](https://github.com/CliMA/EnsembleKalmanProcesses.jl)

The computer model is supplied by the user – it is a forward model, i.e., it takes certain parameters and produces data that can then be compared with the actual observations. We can think of that model as a parameter-to-data map ``G(u): \mathbb{R}^p \rightarrow \mathbb{R}^d``. For example, ``G`` could be a global climate model or a model that predicts the motion of a robot arm. 

The data produced by the forward model are compared to observations $y$, which are assumed to be corrupted by additive noise ``\eta``, such that
```math
y = G(u) + \eta,
```
where the noise ``\eta`` is drawn from a d-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``.

Given knowledge of the  observations ``y``, the forward model ``G(u): \mathbb{R}^p \rightarrow \mathbb{R}^d``, and some information about the noise level such as its size or distribution (but not its value), the inverse problem we want to solve is to find the unknown parameters ``u``.
 
As the name suggests, `CalibrateEmulateSample.jl` breaks this problem into a sequence of three steps: calibration, emulation, and sampling.
A comprehensive treatment of the calibrate-emulate-sample approach to Bayesian inverse problems can be found in [Cleary et al., 2020](https://arxiv.org/pdf/2001.03689.pdf).

## How does it work?

The **calibrate** step of the algorithm consists of an application of [Ensemble Kalman Processes](https://github.com/CliMA/EnsembleKalmanProcesses.jl), that generates input-output pairs in high density around an optimal parameter ``u^*``. This ``u^*`` will be near a mode of the posterior distribution

The **emulate** step takes these pairs and trains a statistical surrogate model (Gaussian process), emulating the forward map ``G``

The **sample** step uses this surrogate in place of ``G`` in a sampling method (Markov chain Monte Carlo) to sample the posterior distribution of ``u``.

`CalibrateEmulateSample.jl` contains the following modules:

Module                       | Purpose
-----------------------------|--------------------------------------------------------
CalibrateEmulateSample.jl    | Pulls in the [Ensemble Kalman Processes](https://github.com/CliMA/EnsembleKalmanProcesses.jl) package
Emulator.jl                  | Emulate: Modular template for emulators
GaussianProcess.jl           | - A Gaussian process emulator
MarkovChainMonteCarlo.jl     | Sample: Modular template for Markov chain Monte Carlo
                             | - Random Walk Metropolis algorithm
                             | - preconditioned Crank-Nicolson algorithm
Utilities.jl                 | Helper functions

**The best way to get started is to have a look at the examples!**
