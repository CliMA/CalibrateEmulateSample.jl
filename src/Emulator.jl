module Emulators

using ..DataContainers
import ..Utilities.encode_with_schedule
import ..Utilities.decode_with_schedule
import ..Utilities.encode_data
import ..Utilities.decode_data
import ..Utilities.encode_structure_matrix
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
function throw_define_mlt()
    throw(ErrorException("Unknown MachineLearningTool defined, please use a known implementation"))
end
function build_models!(mlt, iopairs, input_structure_mats, output_structure_mats, mlt_kwargs...)
    throw_define_mlt()
end
function optimize_hyperparameters!(mlt)
    throw_define_mlt()
end
function predict(mlt, new_inputs; mlt_kwargs...)
    throw_define_mlt()
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
struct ForwardMapWrapper{FT <: Real, VV <: AbstractVector, PD <: ParameterDistribution}
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

"""
$(TYPEDSIGNATURES)

Encode the new data (a `DataContainer`, or matrix where data are columns) representing inputs (`"in"`) or outputs (`"out"`). with the stored and initialized encoder schedule.
"""
function encode_data(
    em_or_fmw::EorFMW,
    data::MorDC,
    in_or_out::AS,
) where {
    AS <: AbstractString,
    MorDC <: Union{AbstractMatrix, DataContainer},
    EorFMW <: Union{Emulator, ForwardMapWrapper},
}
    if isa(data, AbstractMatrix)
        return get_data(encode_with_schedule(get_encoder_schedule(em_or_fmw), DataContainer(data), in_or_out))
    else
        return encode_with_schedule(get_encoder_schedule(em_or_fmw), data, in_or_out)
    end
end

"""
$(TYPEDSIGNATURES)

Encode a new structure matrix in the input space (`"in"`) or output space (`"out"`). with the stored and initialized encoder schedule. 
"""
function encode_structure_matrix(
    em_or_fmw::EorFMW,
    structure_mat,
    in_or_out::AS,
) where {AS <: AbstractString, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return encode_with_schedule(get_encoder_schedule(em_or_fmw), structure_mat, in_or_out)
end


"""
$(TYPEDSIGNATURES)

Decode the new data (a `DataContainer`, or matrix where data are columns) representing inputs (`"in"`) or outputs (`"out"`). with the stored and initialized encoder schedule.
"""
function decode_data(
    em_or_fmw::EorFMW,
    data::MorDC,
    in_or_out::AS,
) where {
    AS <: AbstractString,
    MorDC <: Union{AbstractMatrix, DataContainer},
    EorFMW <: Union{Emulator, ForwardMapWrapper},
}
    if isa(data, AbstractMatrix)
        return get_data(decode_with_schedule(get_encoder_schedule(em_or_fmw), DataContainer(data), in_or_out))
    else
        return decode_with_schedule(get_encoder_schedule(em_or_fmw), data, in_or_out)
    end
end

"""
$(TYPEDSIGNATURES)

Decode a new structure matrix in the input space (`"in"`) or output space (`"out"`). with the stored and initialized encoder schedule. 
"""
function decode_structure_matrix(
    em_or_fmw::EorFMW,
    structure_mat,
    in_or_out::AS,
) where {AS <: AbstractString, EorFMW <: Union{Emulator, ForwardMapWrapper}}
    return decode_with_schedule(get_encoder_schedule(em_or_fmw), structure_mat, in_or_out)
end

"""
$(TYPEDSIGNATURES)

Makes a prediction using the emulator on new inputs (each new inputs given as data columns).
Default is to predict in the decorrelated space.

Return type of N inputs: (in the output space)
  - 1-D: mean [1 x N], cov [1 x N]
  - p-D: mean [p x N], cov N x [p x p] 
"""
function predict(
    emulator::Emulator{FT},
    new_inputs::AM;
    transform_to_real = false,
    mlt_kwargs...,
) where {FT <: AbstractFloat, AM <: AbstractMatrix}
    # Check if the size of new_inputs is consistent with the training data input
    input_dim, output_dim = size(get_io_pairs(emulator), 1)
    encoded_input_dim, encoded_output_dim = size(get_encoded_io_pairs(emulator), 1)

    N_samples = size(new_inputs, 2)

    if size(new_inputs, 1) != input_dim
        throw(
            ArgumentError(
                "Emulator object `io_pairs` and new inputs do not have consistent dimensions, expected $(input_dim), received $(size(new_inputs,1))",
            ),
        )
    end

    # encode the new input data
    encoded_inputs = encode_data(emulator, new_inputs, "in")
    # predict in encoding space
    # returns outputs: [enc_out_dim x n_samples]
    # Scalar-methods uncertainties=variances: [enc_out_dim x n_samples]
    # Vector-methods uncertainties=covariances: [enc_out_dim x enc_out_dim x n_samples)
    encoded_outputs, encoded_uncertainties = predict(get_machine_learning_tool(emulator), encoded_inputs; mlt_kwargs...)

    var_or_cov = (ndims(encoded_uncertainties) == 2) ? "var" : "cov"

    # return decoded or encoded?
    if transform_to_real
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

function forward_map_wrapper(
    forward_map::Function,
    prior::PD,
    input_output_pairs::PairedDataContainer{FT};
    encoder_schedule = nothing,
    encoder_kwargs = NamedTuple(),
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
    else
        @warn "Please note that only the output encoder is used in this implementation. \nThe input encoder will be initialized if provided, but not used during sampling, which is completed in the full parameter space."
    end

    encoder_schedule = create_encoder_schedule(encoder_schedule)
    (encoded_io_pairs, input_structure_mats, output_structure_mats, _, _) =
        initialize_and_encode_with_schedule!(encoder_schedule, input_output_pairs; encoder_kwargs...)

    return ForwardMapWrapper{FT, typeof(encoder_schedule), typeof(prior)}(
        forward_map,
        prior,
        input_output_pairs,
        encoded_io_pairs,
        encoder_schedule,
    )
end



function predict(
    fmw::FMW,
    new_inputs::AM;
    transform_to_real = false,
) where {FMW <: ForwardMapWrapper, AM <: AbstractMatrix}
    # Check if the size of new_inputs is consistent with the training input data
    input_dim, output_dim = size(get_io_pairs(fmw), 1)
    encoded_input_dim, encoded_output_dim = size(get_encoded_io_pairs(fmw), 1)

    N_samples = size(new_inputs, 2)

    if size(new_inputs, 1) != input_dim
        throw(
            ArgumentError(
                "ForwardMapObject `io_pairs` and new inputs do not have consistent dimensions, expected $(input_dim), received $(size(new_inputs,1))",
            ),
        )
    end

    # Scalar-methods uncertainties=variances: [enc_out_dim x n_samples]
    # Vector-methods uncertainties=covariances: [enc_out_dim x enc_out_dim x n_samples)

    # unlike the emulator, the forward map runs in the physical, decoded space. and must be encoded where necessary 
    prior = get_prior(fmw)
    forward_map = get_forward_map(fmw)
    fm_unc = x -> forward_map(transform_unconstrained_to_constrained(prior, x))

    decoded_outputs = reduce(hcat, map(fm_unc, eachcol(new_inputs))) # apply map and return: [out_dim x n_samples]

    var_or_cov = (output_dim == 1) ? "var" : "cov"
    if transform_to_real
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

end
