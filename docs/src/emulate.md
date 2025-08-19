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
    output_structure_matrix = Γy,
    encoder_schedule = encoder_schedule,
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
y, cov = Emulator.predict(emulator, new_inputs)
```
This returns both a mean value and a covariance.

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

Please get in touch with our development team when contributing new statistical emulators, to help us ensure the smoothest interface with any new tools.

