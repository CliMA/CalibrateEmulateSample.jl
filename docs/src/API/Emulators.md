# Emulators

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

```@docs
Emulator
function Emulator(
    ::MachineLearningTool,
    ::PairedDataContainer{FT};
    ::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing},
    ::Bool,
    ::Bool,
    ::Union{AbstractVector{FT}, Nothing},
    ::Bool,
    ::FT,
) where {FT <: AbstractFloat}
optimize_hyperparameters!(::Emulator)
predict
normalize
standardize
reverse_standardize
svd_transform
svd_reverse_transform_mean_cov
```