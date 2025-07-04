# GaussianProcess

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

```@docs
GaussianProcessesPackage
PredictionType
GaussianProcess
GaussianProcess(
    ::GPPkg;
    ::Union{K, KPy, AK, Nothing},
    ::Any,
    ::FT,
    ::PredictionType,
) where {GPPkg <: GaussianProcessesPackage, K <: GaussianProcesses.Kernel, KPy <: PyObject, AK <:AbstractGPs.Kernel, FT <: AbstractFloat}
build_models!(::GaussianProcess{GPJL}, ::PairedDataContainer{FT}, input_structure_matrix, output_structure_matrix) where {FT <: AbstractFloat}
optimize_hyperparameters!(::GaussianProcess{GPJL})
predict(::GaussianProcess{GPJL},  ::AbstractMatrix{FT}) where {FT <: AbstractFloat}
```