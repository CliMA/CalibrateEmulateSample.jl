# RandomFeatures

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

## Scalar interface

```@docs
ScalarRandomFeatureInterface
ScalarRandomFeatureInterface(::Int,::Int)
build_models!(::ScalarRandomFeatureInterface, ::PairedDataContainer{FT}) where {FT <: AbstractFloat}
predict(::ScalarRandomFeatureInterface, ::M) where {M <: AbstractMatrix}
```

## Vector Interface

```@docs
VectorRandomFeatureInterface
VectorRandomFeatureInterface(::Int, ::Int, ::Int)
build_models!(::VectorRandomFeatureInterface, ::PairedDataContainer{FT}) where {FT <: AbstractFloat}
predict(::VectorRandomFeatureInterface, ::M) where {M <: AbstractMatrix}
```

## Other utilities
```@docs
get_rfms
get_fitted_features
get_batch_sizes
get_n_features
get_input_dim
get_output_dim
get_rng
get_diagonalize_input
get_feature_decomposition
get_optimizer_options
optimize_hyperparameters!(::ScalarRandomFeatureInterface) 
optimize_hyperparameters!(::VectorRandomFeatureInterface) 
```
