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
Wrapping a predefined machine learning tool, e.g. a Gaussian process `gauss_proc`, the `Emulator` can then be built:

```julia
emulator = Emulator(
    gauss_proc, 
    input_output_pairs; # optional arguments after this
    obs_noise_cov = Γy,
    normalize_inputs = true,
    standardize_outputs = true,
    standardize_outputs_factors = factor_vector,
    retained_svd_frac = 0.95,
)
```
The optional arguments above relate to the data processing.

## Data processing

Some effects of the following are outlined in a practical setting in the results and appendices of [Howland, Dunbar, Schneider, (2022)](https://doi.org/10.1029/2021MS002735).

### Diagonalization and output dimension reduction

This arises from the optional arguments
- `obs_noise_cov = Γy` (default: `nothing`)
We always use singular value decomposition to diagonalize the output space, requiring output covariance `Γy`. *Why?* If we need to train a $$\mathbb{R}^{10} \to \mathbb{R}^{100}$$ emulator, diagonalization allows us to instead train 100 $$\mathbb{R}^{10} \to \mathbb{R}^{1}$$ emulators (far cheaper).
- `retained_svd_frac = 0.95` (default `1.0`)
Performance is increased further by throwing away less informative output dimensions, if 95% of the information (i.e., variance) is in the first 40 diagonalized output dimensions then setting `retained_svd_frac=0.95` will train only 40 emulators.

!!! note
    Diagonalization is an approximation. It is however a good approximation when the observational covariance varies slowly in the parameter space.
!!! warn
    Severe approximation errors can occur if `obs_noise_cov` is not provided.


### Normalization and standardization

This arises from the optional arguments
- `normalize_inputs = true` (default: `true`)
We normalize the input data in a standard way by centering, and scaling with the empirical covariance
- `standardize_outputs = true` (default: `false`)
- `standardize_outputs_factors = factor_vector` (default: `nothing`)
To help with poor conditioning of the covariance matrix, users can also standardize each ouput dimenion with by a multiplicative factor given by the elements of `factor_vector`

## Modular interface

Each statistical emulator has the following supertype and methods:

```julia
abstract type MachineLearningTool end
function build_models!(mlt, iopairs)
function optimize_hyperparameters!(mlt)
function predict(mlt, new_inputs)
```
Add a new tool as follows:
1. Create `MyMLToolName.jl`, and include "MyMLToolName.jl" in `Emulators.jl`
2. Create a struct `MyMLTool <: MachineLearningTool` 
3. Create these three methods to build, train, and predict with your tool (use `GaussianProcess.jl` as a guide)

!!! note
    The `predict` method currently needs to return both a predicted mean and a predicted (co)variance at new inputs, which are used in the *Sample* stage.

