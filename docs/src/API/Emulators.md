# Emulators

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

```@docs
Emulator
Emulator(::MachineLearningTool, ::PairedDataContainer{FT})
optimize_hyperparameters!(::Emulator)
predict
normalize
standardize
reverse_standardize
svd_transform
svd_reverse_transform_mean_cov
```