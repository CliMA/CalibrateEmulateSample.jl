# CalibrateEmulateSample.jl

|||
|---------------------:|:----------------------------------------------|
| **Documentation**    | [![dev][docs-dev-img]][docs-dev-url]          |
| **Travis Build**     | [![travis][travis-img]][travis-url]           |
| **AppVeyor Build**   | [![appveyor][appveyor-img]][appveyor-url]     |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]        |
| **Bors**             | [![Bors enabled][bors-img]][bors-url]         |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://climate-machine.github.io/CalibrateEmulateSample.jl/dev/

[travis-img]: https://travis-ci.org/climate-machine/CalibrateEmulateSample.jl.svg?branch=master
[travis-url]: https://travis-ci.org/climate-machine/CalibrateEmulateSample.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/c6eykd0w94pmyjt8/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/climate-machine/calibrateemulatesample-jl/branch/master

[codecov-img]: https://codecov.io/gh/climate-machine/CalibrateEmulateSample.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/climate-machine/CalibrateEmulateSample.jl

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/11528


`CalibrateEmulateSample.jl` solves parameter estimation problems using (approximate) Bayesian inversion. It is designed for problems that require running a computer model that is expensive to evaluate, but can also be used for simple models.

The computer model is supplied by the user – it is a forward model, i.e., it takes certain parameters and produces data that can then be compared with the actual observations. We can think of that model as a parameter-to-data map <a href="https://www.codecogs.com/eqnedit.php?latex=G(\theta):&space;\mathbb{R}^p&space;\rightarrow&space;\mathbb{R}^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G(\theta):&space;\mathbb{R}^p&space;\rightarrow&space;\mathbb{R}^d" title="G(\theta): \mathbb{R}^p \rightarrow \mathbb{R}^d" /></a>. For example, <a href="https://www.codecogs.com/eqnedit.php?latex=G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G" title="G" /></a> could be a global climate model or a model that predicts the motion of a robot arm. 

The data produced by the forward model are compared to observations <a href="https://www.codecogs.com/eqnedit.php?latex=y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y" title="y" /></a>, which are assumed to be corrupted by additive noise <a href="https://www.codecogs.com/eqnedit.php?latex=\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\eta" title="\eta" /></a>, such that

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;G(\theta)&space;&plus;&space;\eta," target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;G(\theta)&space;&plus;&space;\eta," title="y = G(\theta) + \eta," /></a>

where the noise <a href="https://www.codecogs.com/eqnedit.php?latex=\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\eta" title="\eta" /></a> is drawn from a d-dimensional Gaussian with distribution <a href="https://www.codecogs.com/eqnedit.php?latex=$N(0,&space;\Gamma_y)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$N(0,&space;\Gamma_y)$" title="$N(0, \Gamma_y)$" /></a>.

**Given knowledge of the  observations <a href="https://www.codecogs.com/eqnedit.php?latex=y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y" title="y" /></a>, the forward model <a href="https://www.codecogs.com/eqnedit.php?latex=G(\theta):&space;\mathbb{R}^p&space;\rightarrow&space;\mathbb{R}^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G(\theta):&space;\mathbb{R}^p&space;\rightarrow&space;\mathbb{R}^d" title="G(\theta): \mathbb{R}^p \rightarrow \mathbb{R}^d" /></a>, and some information about the noise level such as its size or distribution (but not its value), the inverse problem we want to solve is to find the unknown parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a>.**
 
As the name suggests, ```CalibrateEmulateSample.jl``` breaks this problem into a sequence of three steps: calibration, emulation, and sampling.
A comprehensive treatment of the calibrate-emulate-sample approach to Bayesian inverse problems can be found in Cleary et al., 2020: https://arxiv.org/pdf/2001.03689.pdf

In a one-sentence summary, the **calibrate** step of the algorithm consists of an Ensemble Kalman inversion that is used to find good training points for a Gaussian process regression, which in turn is used as a surrogate (**emulator**) of the original forward model <a href="https://www.codecogs.com/eqnedit.php?latex=G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G" title="G" /></a> in the subsequent Markov chain Monte Carlo **sampling** of the posterior distributions of the unknown parameters.


`CalibrateEmulateSample.jl` contains the following modules:

Module                  | Purpose
----------------------- | -------------------------------------------------------
EKI.jl                  | Calibrate – Ensemble Kalman inversion
GPEmulator.jl           | Emulate – Gaussian process regression
MCMC.jl                 | Sample – Markov chain Monte Carlo
GModel.jl               | Forward model G – to be supplied/modified by the user!
Obs.jl                  | Structure to hold observations
ConvenienceFunctions.jl | Helper functions

**The best way to get started is to have a look at the examples!**
