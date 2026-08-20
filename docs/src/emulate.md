# The Emulate stage

Emulation is performed through the construction of an `Emulator` object, which has two components
1. A wrapper for any statistical emulator,
2. Data-processing and dimensionality reduction functionality.

## Typical construction from `Lorenz_example.jl`

First, obtain data in a `PairedDataContainer`, for example, get this from an `EnsembleKalmanProcess` `ekpobj` generated during the `Calibrate` stage, or see the constructor [here](https://github.com/CliMA/EnsembleKalmanProcesses.jl/blob/main/src/DataContainers.jl)
```julia
using CalibrateEmulateSample.Utilities
input_output_pairs = Utilities.get_training_points(ekpobj, 5) # use first 5 iterations as data
```

!!! note "Minibatched calibration"
    The `EnsembleKalmanProcess` can be built with an `ObservationSeries` of `N` samples and a minibatcher with `n` batches. In this case, each iteration fits the ensemble to one of the batches ``B_1, \ldots, B_n`` (``n \leq N``) that form a disjoint partition of statistically similar observations ``y_1, \ldots, y_N \sim \rho``. The batching can aid the calibration stage but does not affect the emulate-sample stages.

    Internally `get_training_points` and `encoder_kwargs_from` reduce outputs/noise from the `EnsembleKalmanProcess` to their per-observation components, thus the emulator is trained on ``(\theta, \mathcal{G}(\theta))`` pairs representative of the distribution ``\rho``. In the Sampling stage the full observation set ``y_1, \ldots, y_N \sim \rho`` is used and batch structure is ignored.

    Warning: using training data with several `(identical-input)->(different-output)` pairs, will induce more sensitivity on the `obs_noise_cov`. It is recommended to ensure it is well estimated (or well-regularized with white noise) to help training; one can also use the `noise_learn=true` flag but this is known to cause slow-downs in training.

Wrapping a predefined machine learning tool, e.g. a Gaussian process `gauss_proc`, the `Emulator` can then be built:

```julia
emulator = Emulator(
    gauss_proc, 
    input_output_pairs; # optional arguments after this
    encoder_schedule = encoder_schedule,
    encoder_kwargs = (; obs_noise_cov = Γy),
)
```
The optional arguments above relate to the data processing, which is described [here](@ref data-proc)

### Emulator Training

The emulator is trained when we combine the machine learning tool and the data into the `Emulator` above. 
For any machine learning tool, hyperparameters are optimized.
```julia
optimize_hyperparameters!(emulator)
```
For some machine learning packages however, this may be completed during construction automatically, and for others this will not. If automatic construction took place, the `optimize_hyperparameters!` line does not perform any new task, so may be safely called. In the Lorenz example, this line learns the hyperparameters of the Gaussian process, which depend on the choice of [kernel](https://clima.github.io/CalibrateEmulateSample.jl/dev/GaussianProcessEmulator/#kernels), and the choice of GP package.
Predictions at new inputs can then be made using
```julia
em_mean, em_cov = Emulator.predict(emulator, new_inputs)
```
This returns both a mean value and a covariance. The emulator is subject to encoding (see [Data processing and Dimension reduction](@ref data-proc)), and so we provide the `encode` and `add_obs_noise_cov` to enable users to predict in different spaces, and with different inflation.
```julia
# produce output in encoded space
em_mean_enc, em_cov_enc = Emulator.predict(emulator, new_inputs; encode="out")

# given encoded inputs, produce outputs in real space, and inflate the emulator uncertainty with observational noise
em_mean, em_and_obs_noise_cov = Emulator.predict(emulator, new_encoded_inputs; encode="in", add_obs_noise_cov=true) 
```

## [Modular interface](@id modular-interface)

Developers may contribute new tools by performing the following
1. Create `MyMLToolName.jl`, and include "MyMLToolName.jl" in `Emulators.jl`
2. Create a struct `MyMLTool <: MachineLearningTool`, containing any arguments or optimizer options 
3. Create the following three methods to build, train, and predict with your tool (use `GaussianProcess.jl` as a guide)
```
build_models!(mlt::MyMLTool, iopairs::PairedDataContainer, input_structure_mats::Dict{Symbol, <:StructureMatrix}, output_structure_mats::Dict{Symbol, <:StructureMatrix}) -> Nothing
optimize_hyperparameters!(mlt::MyMLTool, args...; kwargs...) -> Nothing
function predict(mlt::MyMLTool, new_inputs::Matrix; kwargs...) -> Matrix, Union{Matrix, Array{,3}
```
!!! note "on dimensions of the predict inputs and outputs"
    The `predict` method takes as input, an `input_dim`-by-`N_new` matrix. It return both a predicted mean and a predicted (co)variance at new inputs.
    (i) for scalar-output methods relying on diagonalization, return `output_dim`-by-`N_new` matrices for mean and variance,
    (ii) For vector-output methods, return `output_dim`-by-`N_new` for mean and `output_dim`-by-`output_dim`-by-`N_new` for covariances.

!!! warning "Centralized observational noise addition"
    `predict(mlt::MyMLTool, ...)` must always return the pure latent (noise-free) (co)variance.
    Do not implement your own observational noise inflation — this is handled centrally by the `Emulator`.

Please get in touch with our development team when contributing new statistical emulators, to help us ensure the smoothest interface with any new tools.

