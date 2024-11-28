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
    ::Union{K, KPy, Nothing},
    ::Any,
    ::FT,
    ::PredictionType,
) where {GPPkg <: GaussianProcessesPackage, K <: Kernel, KPy <: PyObject, FT <: AbstractFloat}
build_models!(::GaussianProcess{GPJL}, ::PairedDataContainer{FT}) where {FT <: AbstractFloat}
optimize_hyperparameters!(::GaussianProcess{GPJL})
predict(::GaussianProcess{GPJL},  ::AbstractMatrix{FT}) where {FT <: AbstractFloat}
```