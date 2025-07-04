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