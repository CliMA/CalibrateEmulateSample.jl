module Emulators

using ..DataContainers
using ..Utilities
# to API for encoding
import ..Utilities.encode_data
import ..Utilities.encode_structure_matrix
import ..Utilities.decode_data
import ..Utilities.decode_structure_matrix

using DocStringExtensions
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
using Random

export Emulator, ForwardMapWrapper

export build_models!
export optimize_hyperparameters!
export predict, encode_data, decode_data, encode_structure_matrix, decode_structure_matrix
export get_machine_learning_tool, get_io_pairs, get_encoded_io_pairs, get_encoder_schedule
export get_forward_map, get_prior, forward_map_wrapper
"""
$(TYPEDEF)

Type to dispatch different emulators:

 - GaussianProcess <: MachineLearningTool
"""
abstract type MachineLearningTool end

# include the different <: ML models

include(joinpath("MachineLearningTools", "GaussianProcess.jl")) #for GaussianProcess
include(joinpath("MachineLearningTools", "RandomFeature.jl")) # Random Freatures
# include(joinpath("MachineLearningTools","NeuralNetwork.jl"))
# etc.

# defaults in error, all MachineLearningTools require these functions.
@noinline function throw_define_mlt(mlt)
    throw(ArgumentError("""
Unknown machine learning tool type — $(typeof(mlt)) does not implement the required emulator interface.

Got:
    typeof(mlt) = $(typeof(mlt))

Suggestion:
    Implement `build_models!`, `optimize_hyperparameters!`, and `predict` for your custom type,
    or use a built-in type such as `GaussianProcess` or `ScalarRandomFeatureInterface`.
"""))
end
function build_models!(mlt, iopairs, input_structure_mats, output_structure_mats, mlt_kwargs...)
    throw_define_mlt(mlt)
end
function optimize_hyperparameters!(mlt)
    throw_define_mlt(mlt)
end
function predict(mlt, new_inputs; mlt_kwargs...)
    throw_define_mlt(mlt)
end

# We will define the different emulator types after the general statements

"""
$(TYPEDEF)

Structure used to represent a general emulator, independently of the algorithm used.

# Fields
$(TYPEDFIELDS)
"""
struct Emulator{FT <: AbstractFloat, VV <: AbstractVector}
    "Machine learning tool, defined as a struct of type MachineLearningTool."
    machine_learning_tool::MachineLearningTool
    "original training data"
    io_pairs::PairedDataContainer{FT}
    "encoded training data"
    encoded_io_pairs::PairedDataContainer{FT}
    "Store of the pipeline to encode (/decode) the data"
    encoder_schedule::VV
end

"""
$(TYPEDSIGNATURES)

Return the `MachineLearningTool` (e.g. `GaussianProcess`, `ScalarRandomFeatureInterface`)
stored in `emulator`.
"""
get_machine_learning_tool(emulator::Emulator) = emulator.machine_learning_tool

"""
$(TYPEDSIGNATURES)

Return the original (unencoded) training `PairedDataContainer` stored in `emulator`.
"""
get_io_pairs(emulator::Emulator) = emulator.io_pairs

"""
$(TYPEDSIGNATURES)

Return the encoded training `PairedDataContainer` stored in `emulator`.
"""
get_encoded_io_pairs(emulator::Emulator) = emulator.encoded_io_pairs

"""
$(TYPEDSIGNATURES)

Return the initialised encoder schedule stored in `emulator`.
"""
get_encoder_schedule(emulator::Emulator) = emulator.encoder_schedule


### Forward Map Wrapper

"""
Wrapper that stores an explicit forward map `f` in place of a trained `Emulator`.
When `predict` is called this object evaluates `E_out ∘ f ∘ c(x)`, where `c` is the
constraint transformation from the prior and `E_out` is the output encoder.

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

Prefer the [`forward_map_wrapper`](@ref) factory function for construction — it
builds and initialises the encoder schedule automatically from training data.

$(METHODLIST)
"""
struct ForwardMapWrapper{FT <: Real, VV <: AbstractVector, PD <: ParameterDistribution, NI <: NoiseInjector}
    "function that represents the forward map"
    forward_map::Function
    "a parameter distribution, containing transformations to constrain the forward map inputs"
    prior::PD
    "data used to construct encoder-decoder"
    io_pairs::PairedDataContainer{FT}
    "encoded data"
    encoded_io_pairs::PairedDataContainer{FT}
    "Store of the pipeline to encode (/decode) the data"
    encoder_schedule::VV
    "For lossy encodings, this determines how to inject noise into the null-space upon decoding"
    noise_injector::NI
end
"""
$(TYPEDSIGNATURES)

Return the forward-map function stored in `fmw`.
"""
get_forward_map(fmw::ForwardMapWrapper) = fmw.forward_map

"""
$(TYPEDSIGNATURES)

Return the `ParameterDistribution` prior stored in `fmw`.
"""
get_prior(fmw::ForwardMapWrapper) = fmw.prior

"""
$(TYPEDSIGNATURES)

Return the original (unencoded) training `PairedDataContainer` stored in `fmw`.
"""
get_io_pairs(fmw::ForwardMapWrapper) = fmw.io_pairs

"""
$(TYPEDSIGNATURES)

Return the encoded training `PairedDataContainer` stored in `fmw`.
"""
get_encoded_io_pairs(fmw::ForwardMapWrapper) = fmw.encoded_io_pairs

"""
$(TYPEDSIGNATURES)

Return the initialised encoder schedule stored in `fmw`.
"""
get_encoder_schedule(fmw::ForwardMapWrapper) = fmw.encoder_schedule

"""
$(TYPEDSIGNATURES)

Return the `NoiseInjector` stored in `fmw`, used for null-space noise injection when decoding.
"""
get_noise_injector(fmw::ForwardMapWrapper) = fmw.noise_injector


### Emulator constructors and methods
"""
$(TYPEDSIGNATURES)

Construct an `Emulator` from a machine-learning tool and paired training data, fitting
the encoder schedule and building the underlying model in encoded space.

# Arguments

- `machine_learning_tool`: the `MachineLearningTool` to train (e.g. `GaussianProcess`, `ScalarRandomFeatureInterface`).
- `input_output_pairs`: training data as a `PairedDataContainer`.
- `encoder_schedule` (keyword, default `nothing`): encoding/decoding pipeline passed to
  [`create_encoder_schedule`](@ref). `nothing` builds a default schedule using
  [`decorrelate_sample_cov`](@ref) for both input and output, or
  `(decorrelate_sample_cov(), decorrelate_structure_mat())` when `:obs_noise_cov` is
  present in `encoder_kwargs`. Pass `[]` to disable encoding.
- `encoder_kwargs` (keyword, default `NamedTuple()`): forwarded to
  [`initialize_and_encode_with_schedule!`](@ref).
- Additional keywords are forwarded to the machine-learning tool initialiser.
"""
function Emulator(
    machine_learning_tool::MachineLearningTool,
    input_output_pairs::PairedDataContainer{FT};
    encoder_schedule = nothing,
    encoder_kwargs = NamedTuple(),
    obs_noise_cov = nothing, # temporary
    mlt_kwargs...,
) where {FT <: AbstractFloat}

    if !isnothing(obs_noise_cov)
        if haskey(encoder_kwargs, :obs_noise_cov)
            @warn "Keyword argument `obs_noise_cov=` is deprecated and will be ignored in favor of `encoder_kwargs=(obs_noise_cov=...)`."
        else
            @warn "Keyword argument `obs_noise_cov=` is deprecated. Please use `encoder_kwargs=(obs_noise_cov=...)` instead."
        end
    end

    # [1.] Initializes and performs data encoding schedule
    # Default processing: decorrelate_sample_cov() where no structure matrix provided, and decorrelate_structure_mat() where provided.
    if isnothing(encoder_schedule)
        encoder_schedule = []

        push!(encoder_schedule, (decorrelate_sample_cov(), "in"))
        if haskey(encoder_kwargs, :obs_noise_cov) || !isnothing(obs_noise_cov)
            push!(encoder_schedule, (decorrelate_structure_mat(), "out"))
        else
            push!(encoder_schedule, (decorrelate_sample_cov(), "out"))
        end
    end

    encoder_schedule = create_encoder_schedule(encoder_schedule)
    (encoded_io_pairs, input_structure_mats, output_structure_mats, _, _) = initialize_and_encode_with_schedule!(
        encoder_schedule,
        input_output_pairs;
        obs_noise_cov = obs_noise_cov,
        encoder_kwargs...,
    )

    # build the machine learning tool in the encoded space
    build_models!(machine_learning_tool, encoded_io_pairs, input_structure_mats, output_structure_mats; mlt_kwargs...)
    return Emulator{FT, typeof(encoder_schedule)}(
        machine_learning_tool,
        input_output_pairs,
        encoded_io_pairs,
        encoder_schedule,
    )
end

"""
$(TYPEDSIGNATURES)

Optimizes the hyperparameters in the machine learning tool. Note that some machine learning packages train hyperparameters on construction so this call is not necessary
"""
function optimize_hyperparameters!(emulator::Emulator{FT}, args...; kwargs...) where {FT <: AbstractFloat}
    optimize_hyperparameters!(emulator.machine_learning_tool, args...; kwargs...)
end

function encode_data(
    em_or_fmw::EorFMW,
    data,
    in_or_out::AS,
) where {AS <: AbstractString, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return encode_data(get_encoder_schedule(em_or_fmw), data, in_or_out)
end

function encode_structure_matrix(
    em_or_fmw::EorFMW,
    structure_mat,
    in_or_out::AS,
) where {AS <: AbstractString, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return encode_structure_matrix(get_encoder_schedule(em_or_fmw), structure_mat, in_or_out)
end

function decode_data(
    em_or_fmw::EorFMW,
    data,
    in_or_out::AS,
) where {AS <: AbstractString, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return decode_data(get_encoder_schedule(em_or_fmw), data, in_or_out)
end

function decode_structure_matrix(
    em_or_fmw::EorFMW,
    structure_mat,
    in_or_out::AS,
) where {AS <: AbstractString, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return decode_structure_matrix(get_encoder_schedule(em_or_fmw), structure_mat, in_or_out)
end

"""
$(TYPEDSIGNATURES)

Return emulator predictions (mean and covariance) at `new_inputs`, where each input
is a column of the matrix.

# Arguments

- `emulator`: trained `Emulator` to query.
- `new_inputs`: `[input_dim × N]` matrix of query points.
- `encode` (keyword, default `nothing`): controls which encoding stages are applied.
  Let `Eᵢ`/`Eₒ` be input/output encoders and `G` the model in encoded space:
  - `nothing`      → `Dₒ∘G∘Eᵢ(x)` — standard user-facing call.
  - `"in"`         → `Dₒ∘G(z)` — inputs already encoded as `z = Eᵢx`.
  - `"out"`        → `G∘Eᵢ(x)` — outputs returned in encoded space.
  - `"in_and_out"` → `G(z)` — inputs encoded, outputs in encoded space (used internally by `sample`).
- `add_obs_noise_cov` (keyword, default `false`): when `true`, adds the stored
  observational noise covariance to the returned uncertainty (used internally by `sample`).
- Additional keywords are forwarded to the machine-learning tool `predict` method.

Returns `(mean, cov)` where for `N` inputs:
- 1-D output: `mean` is `[1 × N]`, `cov` is `[1 × N]` variances.
- p-D output: `mean` is `[p × N]`, `cov` is a length-N iterator of `[p × p]` covariance matrices.
"""
function predict(
    emulator::Emulator{FT},
    new_inputs::AM;
    encode = nothing, # maps decoded inputs to decoded outputs
    add_obs_noise_cov = false,
    transform_to_real = nothing,
    mlt_kwargs...,
) where {FT <: AbstractFloat, AM <: AbstractMatrix}

    encode, add_obs_noise_cov = deprecate_transform_to_real(encode, add_obs_noise_cov, transform_to_real)

    # For the logic below
    in_already_encoded = encode ∈ ["in", "in_and_out"]
    out_to_be_decoded = encode ∉ ["out", "in_and_out"]

    # Check if the size of new_inputs is consistent with the training data input
    input_dim, output_dim = size(get_io_pairs(emulator), 1)
    encoded_input_dim, encoded_output_dim = size(get_encoded_io_pairs(emulator), 1)

    check_dim = in_already_encoded ? encoded_input_dim : input_dim
    N_samples = size(new_inputs, 2)

    size(new_inputs, 1) != check_dim && _throw_input_dim_mismatch(size(new_inputs, 1), check_dim; caller = :Emulator)


    # encode the new input data
    if !in_already_encoded
        encoded_inputs = encode_data(emulator, new_inputs, "in")
    else
        encoded_inputs = new_inputs
    end
    # predict in encoding space
    # returns outputs: [enc_out_dim x n_samples]
    # Scalar-methods uncertainties=variances: [enc_out_dim x n_samples]
    # Vector-methods uncertainties=covariances: [enc_out_dim x enc_out_dim x n_samples)
    encoded_outputs, encoded_uncertainties = predict(
        get_machine_learning_tool(emulator),
        encoded_inputs;
        add_obs_noise_cov = add_obs_noise_cov,
        mlt_kwargs...,
    )

    var_or_cov = (ndims(encoded_uncertainties) == 2) ? "var" : "cov"

    # return decoded or encoded?
    if out_to_be_decoded
        decoded_outputs = decode_data(emulator, encoded_outputs, "out")

        decoded_covariances = zeros(eltype(encoded_outputs), output_dim, output_dim, size(encoded_uncertainties)[end])
        if var_or_cov == "var"
            for (i, col) in enumerate(eachcol(encoded_uncertainties))
                decoded_covariances[:, :, i] .= Matrix(decode_structure_matrix(emulator, Diagonal(col), "out"))
            end
        else # == "cov"
            for (i, mat) in enumerate(eachslice(encoded_uncertainties, dims = 3))
                decoded_covariances[:, :, i] .= Matrix(decode_structure_matrix(emulator, mat, "out"))
            end
        end

        if output_dim > 1
            return decoded_outputs, eachslice(decoded_covariances, dims = 3)
        else
            # here the covs are [1 x 1 x samples] just return [1 x samples]
            return decoded_outputs, decoded_covariances[1, :, :]
        end

    else

        encoded_covariances_mat =
            zeros(eltype(encoded_outputs), encoded_output_dim, encoded_output_dim, size(encoded_uncertainties)[end])
        if var_or_cov == "var"
            for (i, col) in enumerate(eachcol(encoded_uncertainties))
                encoded_covariances_mat[:, :, i] = Diagonal(col)
            end
        else # =="cov"
            for (i, mat) in enumerate(eachslice(encoded_uncertainties, dims = 3))
                encoded_covariances_mat[:, :, i] = mat
            end
        end

        if encoded_output_dim > 1
            return encoded_outputs, eachslice(encoded_covariances_mat, dims = 3)
        else
            # here the covs are [1 x 1 x samples] just return [1 x samples]
            return encoded_outputs, encoded_covariances_mat[1, :, :]
        end
    end

end

### Forward Map constructor and methods
"""
$(TYPEDSIGNATURES)

Construct a [`ForwardMapWrapper`](@ref) from an explicit forward map, a prior, and
paired training data. Behaves similarly to the `Emulator` constructor but additionally
requires `prior` so that inputs can be constrained and null-space noise can be injected
when the encoder is lossy.

# Arguments

- `forward_map`: function `F(x)` mapping physical (constrained) parameters to model outputs.
- `prior`: `ParameterDistribution` describing the prior on physical parameters.
- `input_output_pairs`: training data as a `PairedDataContainer`.
- `encoder_schedule` (keyword, default `nothing`): encoding pipeline; see [`Emulator`](@ref) constructor for details.
- `encoder_kwargs` (keyword, default `NamedTuple()`): forwarded to [`initialize_and_encode_with_schedule!`](@ref).
- `noise_injector_threshold` (keyword, default `0.001`): if the variance lost by the
  encoder exceeds this threshold, noise consistent with the prior is injected into the
  null-space when decoding encoded MCMC samples.
- `noise_injector_scaling` (keyword, default `1.0`): multiplicative scale applied to
  injected noise; values below 1.0 can improve robustness for non-Gaussian problems.
"""
function forward_map_wrapper(
    forward_map::Function,
    prior::PD,
    input_output_pairs::PairedDataContainer{FT};
    encoder_schedule = nothing,
    encoder_kwargs = NamedTuple(),
    noise_injector_threshold = 0.001,
    noise_injector_scaling = 1.0,
) where {FT <: Real, PD <: ParameterDistribution}

    # Default processing: decorrelate_sample_cov() where no structure matrix provided, and decorrelate_structure_mat() where provided.
    if isnothing(encoder_schedule)
        encoder_schedule = []

        push!(encoder_schedule, (decorrelate_sample_cov(), "in"))
        if haskey(encoder_kwargs, :obs_noise_cov)
            push!(encoder_schedule, (decorrelate_structure_mat(), "out"))
        else
            push!(encoder_schedule, (decorrelate_sample_cov(), "out"))
        end
    end

    encoder_schedule = create_encoder_schedule(encoder_schedule)
    (encoded_io_pairs, input_structure_mats, output_structure_mats, _, _) =
        initialize_and_encode_with_schedule!(encoder_schedule, input_output_pairs; encoder_kwargs...)

    # As we apply FMW in decoded space, it may be that we need to add additional noise if the encoder is suitably lossy (determined by >`noise_injector_threshold`). We create a noise injector which puts noise in the null space, retaining correlations from the prior. Precompute it here:
    noise_injector = create_noise_injector(encoder_schedule, prior, noise_injector_threshold, noise_injector_scaling)


    return ForwardMapWrapper{FT, typeof(encoder_schedule), typeof(prior), typeof(noise_injector)}(
        forward_map,
        prior,
        input_output_pairs,
        encoded_io_pairs,
        encoder_schedule,
        noise_injector,
    )
end


"""
$(TYPEDSIGNATURES)

Return predictions (mean and covariance) from the `ForwardMapWrapper` at `new_inputs`,
where each input is a column. The forward map runs in the physical (decoded) space and is
encoded/decoded as requested.

# Arguments

- `fmw`: `ForwardMapWrapper` to query.
- `new_inputs`: `[input_dim × N]` matrix of query points.
- `encode` (keyword, default `nothing`): controls encoding stages applied to inputs/outputs.
  Let `Di`/`Eo` be input decoder/output encoder and `G` the forward map in decoded space:
  - `nothing`      → `G(x)` — standard user-facing call.
  - `"in"`         → `G∘Di(z)` — inputs are encoded as `z = Ei(x)`.
  - `"out"`        → `Eo∘G(x)` — outputs returned in encoded space.
  - `"in_and_out"` → `Eo∘G∘Di(z)` — used internally by `sample`.
- `add_obs_noise_cov` (keyword, default `false`): when `true`, adds observational noise
  covariance to the returned uncertainty (used internally by `sample`).

Returns `(mean, cov)` with the same shape conventions as [`predict`](@ref).
"""
function predict(
    fmw::FMW,
    new_inputs::AM;
    encode = nothing, # maps decoded inputs to decoded outputs
    add_obs_noise_cov = false,
    transform_to_real = nothing,
) where {FMW <: ForwardMapWrapper, AM <: AbstractMatrix}

    encode, add_obs_noise_cov = deprecate_transform_to_real(encode, add_obs_noise_cov, transform_to_real)

    # For the logic below
    in_already_encoded = encode ∈ ["in", "in_and_out"]
    out_to_be_decoded = encode ∉ ["out", "in_and_out"]

    # Check if the size of new_inputs is consistent with the training data input
    input_dim, output_dim = size(get_io_pairs(fmw), 1)
    encoded_input_dim, encoded_output_dim = size(get_encoded_io_pairs(fmw), 1)

    check_dim = in_already_encoded ? encoded_input_dim : input_dim
    N_samples = size(new_inputs, 2)

    size(new_inputs, 1) != check_dim &&
        _throw_input_dim_mismatch(size(new_inputs, 1), check_dim; caller = :ForwardMapWrapper)

    prior = get_prior(fmw)
    if in_already_encoded
        # if suitably lossy encoder, we must also inject noise into its null space
        decoded_inputs = decode_and_add_noise(get_noise_injector(fmw), new_inputs)
    else
        decoded_inputs = new_inputs
    end
    # Scalar-methods uncertainties=variances: [enc_out_dim x n_samples]
    # Vector-methods uncertainties=covariances: [enc_out_dim x enc_out_dim x n_samples)

    # unlike the emulator, the forward map runs in the physical, decoded space. and must be encoded where necessary 
    forward_map = get_forward_map(fmw)
    fm_unc = x -> forward_map(transform_unconstrained_to_constrained(prior, x))

    decoded_outputs = reduce(hcat, map(fm_unc, eachcol(decoded_inputs))) # apply map and return: [out_dim x n_samples]

    var_or_cov = (output_dim == 1) ? "var" : "cov"
    if out_to_be_decoded
        # uncertainty returned is just `I` in encoded space
        decoded_cov = Matrix(decode_structure_matrix(fmw, I(output_dim), "out"))

        decoded_covariances = zeros(eltype(decoded_outputs), output_dim, output_dim, size(decoded_outputs, 2))
        for i in 1:size(decoded_covariances, 3)
            decoded_covariances[:, :, i] .= decoded_cov
        end

        if output_dim > 1
            return decoded_outputs, eachslice(decoded_covariances, dims = 3)
        else
            # here the covs are [1 x 1 x samples] just return [1 x samples]
            return decoded_outputs, decoded_covariances[1, :, :]
        end

    else # We encode
        encoded_outputs = Matrix(encode_data(fmw, decoded_outputs, "out"))
        encoded_output_dim = size(encoded_outputs, 1)
        encoded_cov = I(encoded_output_dim)

        encoded_covariances_mat =
            zeros(eltype(encoded_outputs), encoded_output_dim, encoded_output_dim, size(encoded_outputs, 2))
        for i in 1:size(encoded_covariances_mat, 3)
            encoded_covariances_mat[:, :, i] = encoded_cov
        end

        if encoded_output_dim > 1
            return encoded_outputs, eachslice(encoded_covariances_mat, dims = 3)
        else
            # here the covs are [1 x 1 x samples] just return [1 x samples]
            return encoded_outputs, encoded_covariances_mat[1, :, :]
        end
    end
end


## Error helpers

@noinline function _throw_input_dim_mismatch(actual::Int, expected::Int; caller::Symbol)
    throw(DimensionMismatch("""
$caller: new_inputs row count does not match the expected input dimension.

Expected:
    size(new_inputs, 1) == $expected

Got:
    size(new_inputs, 1) = $actual

Suggestion:
    Pass new_inputs with $expected rows (one per input dimension).
    If inputs are already in the encoded space, set `in_already_encoded = true`.
"""))
end

### Deprecated keywords

function deprecate_transform_to_real(encode, add_obs_noise_cov, transform_to_real)
    if !isnothing(transform_to_real)
        @warn(
            """`transform_to_real` keyword is deprecated. Please use the `encode` and `add_obs_noise_cov` keywords instead.
                                             
Recommended usage for users is now set by default as:
 - `encode=nothing`, `add_obs_noise_cov=false`
This behaviour takes in non-encoded inputs, and returns non-encoded outputs. It gives only the uncertainty from the Machine Learning Tool (not inflated by observational noise)
               
This simulation will continue with the old behavior:
 - `transform_to_real=true` replaced with `encode=nothing, add_obs_noise_cov=true`
 - `transform_to_real=false` replaced with `encode="out", add_obs_noise_cov=true`
    """,
            maxlog = 1
        )

        # modify kwargs
        add_onc = true
        enc = transform_to_real ? nothing : "out"
        return enc, add_onc
    else
        return encode, add_obs_noise_cov
    end
end



end
