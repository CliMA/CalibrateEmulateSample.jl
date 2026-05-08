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
function throw_define_mlt(mlt)
    throw(
        ErrorException(
            "Unknown MachineLearningTool defined, please use a known implementation. Please check all methods are defined for the MLT received: \n $mlt",
        ),
    )
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

Gets the `machine_learning_tool` field of the `Emulator`
"""
get_machine_learning_tool(emulator::Emulator) = emulator.machine_learning_tool

"""
$(TYPEDSIGNATURES)

Gets the `io_pairs` field of the `Emulator`
"""
get_io_pairs(emulator::Emulator) = emulator.io_pairs

"""
$(TYPEDSIGNATURES)

Gets the `encoded_io_pairs` field of the `Emulator`
"""
get_encoded_io_pairs(emulator::Emulator) = emulator.encoded_io_pairs

"""
$(TYPEDSIGNATURES)

Gets the `encoder_schedule` field of the `Emulator`
"""
get_encoder_schedule(emulator::Emulator) = emulator.encoder_schedule


### Forward Map Wrapper

"""
$(TYPEDEF)
This can replace an `Emulator`, but stores the original forward map. The forward map must be definable as a function `f`. To apply `f` properly this object also builds and stores an encoder `E`  and a parameter distribution (i.e. the prior) containing physical constraints `c`.

When predict() is called this map will call `E_{out}∘f∘c(x)`

# Fields
$(TYPEDFIELDS)

# Constructors:
- `forward_map_wrapper(forward_map, prior, input_output_pairs; encoder_schedule=nothing, encoder_kwargs=NamedTuple())`
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

Gets the `forward_map` field of the `ForwardMapWrapper`
"""
get_forward_map(fmw::ForwardMapWrapper) = fmw.forward_map

"""
$(TYPEDSIGNATURES)

Gets the `prior` field of the `ForwardMapWrapper`
"""
get_prior(fmw::ForwardMapWrapper) = fmw.prior

"""
$(TYPEDSIGNATURES)

Gets the `io_pairs` field of the `ForwardMapWrapper`
"""
get_io_pairs(fmw::ForwardMapWrapper) = fmw.io_pairs

"""
$(TYPEDSIGNATURES)

Gets the `encoded_io_pairs` field of the `ForwardMapWrapper`
"""
get_encoded_io_pairs(fmw::ForwardMapWrapper) = fmw.encoded_io_pairs

"""
$(TYPEDSIGNATURES)

Gets the `encoder_schedule` field of the `ForwardMapWrapper`
"""
get_encoder_schedule(fmw::ForwardMapWrapper) = fmw.encoder_schedule

"""
$(TYPEDSIGNATURES)

Gets the `noise_injector` field of the `ForwardMapWrapper`
"""
get_noise_injector(fmw::ForwardMapWrapper) = fmw.noise_injector


### Emulator constructors and methods
"""
$(TYPEDSIGNATURES)

Constructor of the Emulator object,

Positional Arguments
 - `machine_learning_tool`: the selected machine learning tool object (e.g. Gaussian process / Random feature interface)
 - `input_output_pairs`: the paired input-output data points stored in a `PairedDataContainer`

Keyword Arguments 
 -  `encoder_schedule`[=`nothing`]: the schedule of data encoding/decoding. This will be passed into the method `create_encoder_schedule` internally. `nothing` sets sets a default schedule `[(decorrelate_sample_cov(), "in_and_out")]`, or `[(decorrelate_sample_cov(), "in"), (decorrelate_structure_mat(), "out")]` if an `encoder_kwargs` has a key `:obs_noise_cov`. Pass `[]` for no encoding.
 - `encoder_kwargs`[=`NamedTuple()`]: a Dict or NamedTuple with keyword arguments to be passed to `initialize_and_encode_with_schedule!`
Other keywords are passed to the machine learning tool initialization
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

Makes a prediction using the emulator on new inputs (each new inputs given as data columns).

Keyword args
- `encode` [=`nothing`]: For the input encoder `Eᵢ`, and output decoder `Dₒ` stored in the emulator, we have learnt a predict method method `G` in the encoded space. Interpret the keyword as follows:
    - `nothing`     : applies Dₒ∘G∘Eᵢ(x) (nothing is encoded) - most common for user interaction
    - `"in"`        : applies Dₒ∘G(z) (the inputs are provided as encoded (z=Eᵢx))
    - `"out"`       : applies G∘Eᵢ(x) (the outputs are returned as encoded)
    - `"in_and_out"`: applies G(z) (inputs (z=Eᵢx) and outputs are both encoded) - internally called by `Sample` method
- `add_obs_noise_cov`[=`false`]: When returning the prediction covariance, whether to add the observational noise
    - `false`: Only return the uncertainty given by the machine learning tool - most common for user emulator validation
    - `true` : Return the sum of emulator and observational uncertainty - internally called by `Sample` method
- All other kwargs are passed into the machine learning tool.

Return type of N inputs: (in the output space)
  - 1-D: mean [1 x N], cov [1 x N]
  - p-D: mean [p x N], cov N x [p x p] 
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

    if size(new_inputs, 1) != check_dim
        throw(
            ArgumentError(
                "Emulator object `io_pairs` (resp. `encoded_io_pairs`) and new inputs do not have consistent dimensions, expected $(check_dim), received $(size(new_inputs,1))",
            ),
        )
    end


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

Constructor of the `ForwardMapWrapper` object. Behaves similarly to constructing the `Emulator` but additionally requires the `prior` parameter distribution.

Positional Arguments
 - `forward_map`: a function that represents the forward map `F(x)` mapping physical parameters (in constrained space) to outputs
 - `prior`: a `ParameterDistribution` object describing the prior on the physical parameters.
 - `input_output_pairs`: the paired input-output data points stored in a `PairedDataContainer`

Keyword Arguments 
 -  `encoder_schedule`[=`nothing`]: the schedule of data encoding/decoding. This will be passed into the method `create_encoder_schedule` internally. `nothing` sets sets a default schedule `[(decorrelate_sample_cov(), "in_and_out")]`, or `[(decorrelate_sample_cov(), "in"), (decorrelate_structure_mat(), "out")]` if an `encoder_kwargs` has a key `:obs_noise_cov`. Pass `[]` for no encoding.
 - `encoder_kwargs`[=`NamedTuple()`]: a Dict or NamedTuple with keyword arguments to be passed to `initialize_and_encode_with_schedule!`
 - `noise_injector_threshold`[=`0.001`]: A threshold to implementing noise injection when decoding from an lossily encoded space. If the variance loss due to encoding is `>noise_injector_threshold` then additional noise is added to the null-space (consistent with the prior correlation structure).
 - `noise_injector_scaling`[=`1.0`]: a multiplicative scaling that is applied to injected noise samples. (1.0 is "consistent" with Gaussian theory, but may cause instability in Non-Gaussian problems and can be reduced)
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

Makes a prediction using the ForwardMapWrapper on new inputs (each new inputs given as data columns).

Keyword args
- `encode` [=`nothing`]: For the output encoder `Eₒ`, and input decoder `Dᵢ` stored in the `ForwardMapWrapper`, we have provided the forward map `G` in the decoded space. Interpret the keyword as follows:
    - `nothing`     : applies G(x) (nothing is encoded) - most common for user interaction
    - `"in"`        : applies G∘Dᵢ(z) (the inputs are provided as encoded (x=Dᵢz))
    - `"out"`       : applies Eₒ∘G(x) (the outputs are returned as encoded)
    - `"in_and_out"`: applies Eₒ∘G∘Dᵢ(z) (inputs (x=Dᵢz) and outputs are both encoded) - internally called by `Sample` method
- `add_obs_noise_cov`[=`false`]: When returning the prediction covariance, whether to add the observational noise
    - `false`: Only return the uncertainty given by the machine learning tool - most common for user emulator validation
    - `true` : Return the sum of emulator and observational uncertainty - internally called by `Sample` method
- All other kwargs are passed into the machine learning tool.

Return type of N inputs: (in the output space)
  - 1-D: mean [1 x N], cov [1 x N]
  - p-D: mean [p x N], cov N x [p x p] 
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

    if size(new_inputs, 1) != check_dim
        throw(
            ArgumentError(
                "ForwardMapWrapper object `io_pairs` (resp. `encoded_io_pairs`) and new inputs do not have consistent dimensions, expected $(check_dim), received $(size(new_inputs,1))",
            ),
        )
    end

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
