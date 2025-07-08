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

export Emulator

export calculate_normalization
export build_models!
export optimize_hyperparameters!
export predict, encode_data, decode_data, encode_structure_matrix, decode_structure_matrix
export get_machine_learning_tool, get_io_pairs, get_encoded_io_pairs, get_encoder_schedule
"""
$(TYPEDEF)

Type to dispatch different emulators:

 - GaussianProcess <: MachineLearningTool
"""
abstract type MachineLearningTool end

# include the different <: ML models
include("GaussianProcess.jl") #for GaussianProcess
include("RandomFeature.jl")
# include("NeuralNetwork.jl")
# etc.

# defaults in error, all MachineLearningTools require these functions.
function throw_define_mlt()
    throw(ErrorException("Unknown MachineLearningTool defined, please use a known implementation"))
end
function build_models!(mlt, iopairs, mlt_kwargs...)
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

Gets the `encoder_schedul` field of the `Emulator`
"""
get_encoder_schedule(emulator::Emulator) = emulator.encoder_schedule

# Constructor for the Emulator Object
"""
$(TYPEDSIGNATURES)

Constructor of the Emulator object,

Positional Arguments
 - `machine_learning_tool`: the selected machine learning tool object (e.g. Gaussian process / Random feature interface)
 - `input_output_pairs`: the paired input-output data points stored in a `PairedDataContainer`

Keyword Arguments 
 -  `encoder_schedule`[=`nothing`]: the schedule of data encoding/decoding. This will be passed into the method `create_encoder_schedule` internally. `nothing` sets sets a default schedule `(decorrelate_samples_cov(), "in_and_out")`. Pass `[]` for no encoding.
 - `input_structure_matrix`[=`nothing`]: Some encoders make use of an input structure (e.g., the prior covariance matrix). Particularly useful for few samples. `nothing` sets a default `I(input_dim)`
 - `output_structure_matrix` [=`nothing`] Some encoders make use of an input structure (e.g., the prior covariance matrix). Particularly useful for few samples. `nothing` sets a default `I(input_dim)`
Other keywords are passed to the machine learning tool initialization
"""
function Emulator(
    machine_learning_tool::MachineLearningTool,
    input_output_pairs::PairedDataContainer{FT};
    encoder_schedule = nothing,
    input_structure_matrix::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing} = nothing,
    output_structure_matrix::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing} = nothing,
    obs_noise_cov = nothing, # temporary 
    mlt_kwargs...,
) where {FT <: AbstractFloat}

    # For Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)

    if !isnothing(obs_noise_cov) && isnothing(output_structure_matrix)
        @warn(
            "Keyword `obs_noise_cov=` is now deprecated, and replaced with `output_structure_matrix`. \n Continuing by setting `output_structure_matrix=obs_noise_cov`."
        )
        output_structure_matrix = obs_noise_cov
    elseif !isnothing(obs_noise_cov) && !isnothing(output_structure_matrix)
        @warn(
            "Keyword `obs_noise_cov=` is now deprecated and will be ignored. \n Continuing with value of `output_structure_matrix=`"
        )
    end

    input_structure_mat = if isnothing(input_structure_matrix)
        Diagonal(FT.(ones(input_dim)))
    elseif isa(input_structure_matrix, UniformScaling)
        input_structure_matrix(input_dim)
    else
        input_structure_matrix
    end

    output_structure_mat = if isnothing(output_structure_matrix)
        Diagonal(FT.(ones(output_dim)))
    elseif isa(output_structure_matrix, UniformScaling)
        output_structure_matrix(output_dim)
    else
        output_structure_matrix
    end

    # [1.] Initializes and performs data encoding schedule
    # Default processing: decorrelate_sample_cov() where no structure matrix provided, and decorrelate_structure_mat() where provided.
    if isnothing(encoder_schedule)
        encoder_schedule = []
        if isnothing(input_structure_matrix)
            push!(encoder_schedule, (decorrelate_sample_cov(), "in"))
        else
            push!(encoder_schedule, (decorrelate_structure_mat(), "in"))
        end
        if isnothing(output_structure_matrix)
            push!(encoder_schedule, (decorrelate_sample_cov(), "out"))
        else
            push!(encoder_schedule, (decorrelate_structure_mat(), "out"))
        end
    end

    enc_schedule = create_encoder_schedule(encoder_schedule)
    (encoded_io_pairs, encoded_input_structure_mat, encoded_output_structure_mat) =
        encode_with_schedule!(enc_schedule, input_output_pairs, input_structure_mat, output_structure_mat)

    # build the machine learning tool in the encoded space
    build_models!(
        machine_learning_tool,
        encoded_io_pairs,
        encoded_input_structure_mat,
        encoded_output_structure_mat;
        mlt_kwargs...,
    )
    return Emulator{FT, typeof(enc_schedule)}(machine_learning_tool, input_output_pairs, encoded_io_pairs, enc_schedule)
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
    emulator::Emulator,
    data::MorDC,
    in_or_out::AS,
) where {AS <: AbstractString, MorDC <: Union{AbstractMatrix, DataContainer}}
    if isa(data, AbstractMatrix)
        return get_data(encode_with_schedule(get_encoder_schedule(emulator), DataContainer(data), in_or_out))
    else
        return encode_with_schedule(get_encoder_schedule(emulator), data, in_or_out)
    end
end

"""
$(TYPEDSIGNATURES)

Encode a new structure matrix in the input space (`"in"`) or output space (`"out"`). with the stored and initialized encoder schedule. 
"""
function encode_structure_matrix(
    emulator::Emulator,
    structure_mat::USorM,
    in_or_out::AS,
) where {AS <: AbstractString, USorM <: Union{UniformScaling, AbstractMatrix}}
    return encode_with_schedule(get_encoder_schedule(emulator), structure_mat, in_or_out)
end


"""
$(TYPEDSIGNATURES)

Decode the new data (a `DataContainer`, or matrix where data are columns) representing inputs (`"in"`) or outputs (`"out"`). with the stored and initialized encoder schedule.
"""
function decode_data(
    emulator::Emulator,
    data::MorDC,
    in_or_out::AS,
) where {AS <: AbstractString, MorDC <: Union{AbstractMatrix, DataContainer}}
    if isa(data, AbstractMatrix)
        return get_data(decode_with_schedule(get_encoder_schedule(emulator), DataContainer(data), in_or_out))
    else
        return decode_with_schedule(get_encoder_schedule(emulator), data, in_or_out)
    end
end

"""
$(TYPEDSIGNATURES)

Decode a new structure matrix in the input space (`"in"`) or output space (`"out"`). with the stored and initialized encoder schedule. 
"""
function decode_structure_matrix(
    emulator::Emulator,
    structure_mat::USorM,
    in_or_out::AS,
) where {AS <: AbstractString, USorM <: Union{UniformScaling, AbstractMatrix}}
    return decode_with_schedule(get_encoder_schedule(emulator), structure_mat, in_or_out)
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
    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension.
    # un-encoded data to get dimensions
    input_dim, output_dim = size(get_io_pairs(emulator), 1)
    encoded_input_dim, encoded_output_dim = size(get_encoded_io_pairs(emulator), 1)

    N_samples = size(new_inputs, 2)

    if size(new_inputs, 1) != input_dim
        throw(
            ArgumentError(
                "Emulator object and input observations do not have consistent dimensions, expected $(input_dim), received $(size(new_inputs,1))",
            ),
        )
    end

    # encode the new input data
    encoded_inputs = encode_data(emulator, new_inputs, "in")
    # predict in encoding space
    # returns outputs: [enc_out_dim x n_samples]
    # Scalar-methods uncertainties=variances: [enc_out_dim x n_samples]
    # Vector-methods uncertainties=covariances: [enc_out_dim x enc_out_dim x n_samples)
    encoded_outputs, encoded_uncertainties = predict(get_machine_learning_tool(emulator), encoded_inputs, mlt_kwargs...)

    var_or_cov = (ndims(encoded_uncertainties) == 2) ? "var" : "cov"

    # return decoded or encoded?
    if transform_to_real
        decoded_outputs = decode_data(emulator, encoded_outputs, "out")

        decoded_covariances = zeros(eltype(encoded_outputs), output_dim, output_dim, size(encoded_uncertainties)[end])
        if var_or_cov == "var"
            for (i, col) in enumerate(eachcol(encoded_uncertainties))
                decoded_covariances[:, :, i] .= decode_structure_matrix(emulator, Diagonal(col), "out")
            end
        else # == "cov"
            for (i, mat) in enumerate(eachslice(encoded_uncertainties, dims = 3))
                decoded_covariances[:, :, i] .= decode_structure_matrix(emulator, mat, "out")
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

end
