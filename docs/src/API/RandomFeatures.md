# RandomFeatures

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```
## Kernel and Covariance structure
```@docs
OneDimFactor
DiagonalFactor
CholeskyFactor
LowRankFactor
HierarchicalLowRankFactor
SeparableKernel
NonseparableKernel
calculate_n_hyperparameters
hyperparameters_from_flat
build_default_prior
```

## Scalar interface

```@docs
ScalarRandomFeatureInterface
ScalarRandomFeatureInterface(::Int,::Int)
build_models!(::ScalarRandomFeatureInterface, ::PairedDataContainer{FT}, input_structure_mats, output_structure_mats) where {FT <: AbstractFloat}
predict(::ScalarRandomFeatureInterface, ::M) where {M <: AbstractMatrix}
```

## Vector Interface

```@docs
VectorRandomFeatureInterface
VectorRandomFeatureInterface(::Int, ::Int, ::Int)
build_models!(::VectorRandomFeatureInterface, ::PairedDataContainer{FT}, input_structure_mats, output_structure_mats) where {FT <: AbstractFloat}
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
EKP.get_rng
get_kernel_structure
get_feature_decomposition
get_optimizer_options
optimize_hyperparameters!(::ScalarRandomFeatureInterface) 
optimize_hyperparameters!(::VectorRandomFeatureInterface) 
shrinkage_cov
```
