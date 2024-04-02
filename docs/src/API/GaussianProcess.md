# GaussianProcess

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

```@docs
GaussianProcessesPackage
PredictionType
GaussianProcess
GaussianProcess(package::GPPkg)  where {GPPkg <: GaussianProcessesPackage}
build_models!(::GaussianProcess{GPJL}, ::PairedDataContainer{FT}) where {FT <: AbstractFloat}
optimize_hyperparameters!(::GaussianProcess{GPJL})
predict(::GaussianProcess{GPJL},  ::AbstractMatrix{FT}) where {FT <: AbstractFloat}
```