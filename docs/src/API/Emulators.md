# Emulators

```@meta
CurrentModule = CalibrateEmulateSample.Emulators
```

```@docs
Emulator
optimize_hyperparameters!(::Emulator)
Emulator(::MachineLearningTool, ::PairedDataContainer{FT}) where {FT <: AbstractFloat}
predict
encode_data
decode_data
encode_structure_matrix
decode_structure_matrix
```

## Forward map wrapper

```@docs
ForwardMapWrapper
forward_map_wrapper
```

## Getter functions

```@docs
get_machine_learning_tool
get_io_pairs
get_encoded_io_pairs
get_encoder_schedule
get_forward_map
get_prior
```