# CalibrateEmulateSample.jl

`CalibrateEmulateSample.jl` solves parameter estimation problems using (approximate) Bayesian inversion. It is designed for problems that require running a computer model that is expensive to evaluate, but can also be used for simple models.

The computer model is supplied by the user – it is a forward model, i.e., it takes certain parameters and produces data that can then be compared with the actual observations. We can think of that model as a parameter-to-data map ``G(u): \mathbb{R}^p \rightarrow \mathbb{R}^d``. For example, ``G`` could be a global climate model or a model that predicts the motion of a robot arm. 

The data produced by the forward model are compared to observations $y$, which are assumed to be corrupted by additive noise ``\eta``, such that
```math
\begin{equation}
y = G(u) + \eta
\end{equation},
```
where the noise ``\eta`` is drawn from a d-dimensional Gaussian with distribution ``\mathcal{N}(0, \Gamma_y)``.

Given knowledge of the  observations ``y``, the forward model ``G(u): \mathbb{R}^p \rightarrow \mathbb{R}^d``, and some information about the noise level such as its size or distribution (but not its value), the inverse problem we want to solve is to find the unknown parameters ``u``.
 
As the name suggests, `CalibrateEmulateSample.jl` breaks this problem into a sequence of three steps: calibration, emulation, and sampling.
A comprehensive treatment of the calibrate-emulate-sample approach to Bayesian inverse problems can be found in [Cleary et al., 2020](https://arxiv.org/pdf/2001.03689.pdf).

In a one-sentence summary, the **calibrate** step of the algorithm consists of an Ensemble Kalman inversion that is used to find good training points for a Gaussian process regression, which in turn is used as a surrogate (**emulator**) of the original forward model ``G`` in the subsequent Markov chain Monte Carlo **sampling** of the posterior distributions of the unknown parameters.


`CalibrateEmulateSample.jl` contains the following modules:

Module                                      | Purpose
--------------------------------------------|--------------------------------------------------------
EnsembleKalmanProcesses.jl                  | Calibrate – Ensemble Kalman inversion
GaussianProcessEmulator.jl                  | Emulate – Gaussian process regression
MarkovChainMonteCarlo.jl                    | Sample – Markov chain Monte Carlo
Observations.jl                             | Structure to hold observations
Utilities.jl                                | Helper functions

**The best way to get started is to have a look at the examples!**
