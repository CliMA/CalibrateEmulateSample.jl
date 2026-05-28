# RandomFeatures

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```
## Kernel and Covariance structure
```@docs
CovarianceStructureType
OneDimFactor
DiagonalFactor
CholeskyFactor
LowRankFactor
HierarchicalLowRankFactor
SeparableKernel
NonseparableKernel
cov_structure_from_string
calculate_n_hyperparameters
hyperparameters_from_flat
build_default_prior
get_eps
get_input_cov_structure
get_output_cov_structure
get_cov_structure
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
nice_cov
```
