# Emulators

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

```@docs
Emulator
optimize_hyperparameters!(::Emulator)
Emulator(::MachineLearningTool, ::PairedDataContainer{FT}) where {FT <: AbstractFloat}
predict
normalize
standardize
reverse_standardize
svd_transform
svd_reverse_transform_mean_cov
```