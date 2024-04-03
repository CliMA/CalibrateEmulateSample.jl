# The Calibrate stage

Calibration of the computer model entails finding an optimal parameter ``\theta^*`` that maximizes the posterior probability

```math
\rho(\theta\vert y, \Gamma_y) = \dfrac{e^{-\mathcal{L}(\theta; y)}}{Z(y\vert\Gamma_y)}\rho_{\mathrm{prior}}(\theta), \qquad \mathcal{L}(\theta, y) = \langle \mathcal{G}(\theta) - y \, , \, \Gamma_y^{-1} \left ( \mathcal{G}(\theta) - y \right ) \rangle,
```
where ``\mathcal{L}`` is the loss or negative log-likelihood, ``Z(y\vert\Gamma)`` is a normalizing constant, ``y`` represents the data, ``\Gamma_y`` is the noise covariance matrix and ``\rho_{\mathrm{prior}}(\theta)`` is the prior density. Calibration is performed using [ensemble Kalman processes](https://github.com/CliMA/EnsembleKalmanProcesses.jl), which generate input-output pairs ``\{\theta, \mathcal{G}(\theta)\}`` in high density from the prior initial guess to the found optimal parameter ``\theta^*``. These input-output pairs are then used as the data to train an emulator of the forward model ``\mathcal{G}``.

## Ensemble Kalman Processes

Calibration can be performed using different ensemble Kalman processes: ensemble Kalman inversion ([Iglesias et al, 2013](http://dx.doi.org/10.1088/0266-5611/29/4/045001)), ensemble Kalman sampler ([Garbuno-Inigo et al, 2020](https://doi.org/10.1137/19M1251655)), and unscented Kalman inversion ([Huang et al, 2022](https://doi.org/10.1016/j.jcp.2022.111262)). All algorithms are implemented in [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl). Documentation of each algorithm is available in the EnsembleKalmanProcesses [docs](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/).

## Typical construction of the EnsembleKalmanProcess

Documentation on how to construct an EnsembleKalmanProcess from the computer model and the data can be found in the EnsembleKalmanProcesses [docs](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/ensemble_kalman_inversion/).

## Julia-free forward model

One draw of our approach is that it does not require the forward map to be written in Julia. To aid construction of such a workflow, EnsembleKalmanProcesses.jl provides a documented example of a BASH workflow for the [sinusoid problem](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/examples/sinusoid_example_toml/), with source code [here](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples/SinusoidInterface). The forward map interacts with the calibration tools (EKP) only though TOML file reading an writing, and thus can be written in any language; for example, to be used with [slurm HPC scripts](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/examples/ClimateMachine_example/), with source code [here](https://github.com/CliMA/EnsembleKalmanProcesses.jl/tree/main/examples/ClimateMachine).

