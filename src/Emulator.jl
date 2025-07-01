module Emulators

using ..DataContainers

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
export predict

"""
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

Structure used to represent a general emulator, independently of the algorithm used.

# Fields
$(DocStringExtensions.TYPEDFIELDS)
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

get_machine_learning_tool(emulator::Emulator) = emulator.machine_learning_tool
get_io_pairs(emulator::Emulator) = emulator.io_pairs
get_encoded_io_pairs(emulator::Emulator) = emulator.encoded_io_pairs
get_encoder_schedule(emulator::Emulator) = emulator.encoder_schedule

# Constructor for the Emulator Object
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Positional Arguments
 - `machine_learning_tool` ::MachineLearningTool,
 - `input_output_pairs` ::PairedDataContainer
Keyword Arguments 
 
"""
function Emulator(
    machine_learning_tool::MachineLearningTool,
    input_output_pairs::PairedDataContainer{FT};
    encoder_schedule = nothing,
    input_structure_matrix::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing} = nothing,
    output_structure_matrix::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing} = nothing,
    mlt_kwargs...,
) where {FT <: AbstractFloat}

    # For Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)

    input_structure_mat = if isnothing(input_structure_matrix)
        Diagonal(FT.(ones(input_dim)))
    elseif isa(input_structure_matrix, UniformScaling)
        Diagonal(input_structure_matrix(input_dim))
    else
        input_structure_matrix
    end
    
    output_structure_mat = if isnothing(output_structure_matrix)
        Diagonal(FT.(ones(output_dim)))
    elseif isa(output_structure_matrix, UniformScaling)
        Diagonal(output_structure_matrix(output_dim))
    else
        output_structure_matrix
    end

    # [1.] Initializes and performs data encoding schedule
    if !isnothing(encoder_schedule)
        enc_schedule = create_encoder_schedule(encoder_schedule)
        (encoded_io_pairs, encoded_input_structure_mat, encoded_output_structure_mat) =
            encode_with_schedule(
                enc_schedule,
                input_output_pairs,
                input_structure_mat,
                output_structure_mat,
            )
    else 
        (encoded_io_pairs, encoded_input_structure_mat, encoded_output_structure_mat) = (input_output_pairs, input_structure_mat,  output_structure_mat)
        enc_schedule = []
    end    
    # build the machine learning tool in the encoded space
    build_models!(machine_learning_tool, encoded_io_pairs, encoded_input_structure_mat, encoded_output_structure_mat; mlt_kwargs...)
    return Emulator{FT, typeof(enc_schedule)}(
        machine_learning_tool,
        input_output_pairs,
        encoded_io_pairs,
        enc_schedule,
    )
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimizes the hyperparameters in the machine learning tool.
"""
function optimize_hyperparameters!(emulator::Emulator{FT}, args...; kwargs...) where {FT <: AbstractFloat}
    optimize_hyperparameters!(emulator.machine_learning_tool, args...; kwargs...)
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes a prediction using the emulator on new inputs (each new inputs given as data columns).
Default is to predict in the decorrelated space.
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
    encoded_input_dim, encoded_output_dim = size(get_io_pairs(emulator), 1)
    
    N_samples = size(new_inputs, 2)
    
    if !(size(new_inputs, 1) == input_dim)
        throw(
            ArgumentError(
                "Emulator object and input observations do not have consistent dimensions, expected $(input_dim), received $(size(new_inputs,1))",
            ),
        )
    end

    # encode the new input data
    encoder_schedule = get_encoder_schedule(emulator)
    encoded_inputs = encode_with_schedule(encoder_schedule, DataContainer(new_inputs), "in")
    # predict in encoding space
    # returns outputs: [enc_out_dim x n_samples]
    # Scalar-methods uncertainties=variances: [enc_out_dim x n_samples]
    # Vector-methods uncertainties=covariances: [enc_out_dim x enc_out_dim x n_samples)
    encoded_outputs, encoded_uncertainties = predict(get_machine_learning_tool(emulator), get_data(encoded_inputs), mlt_kwargs...)
    var_or_cov = (ndims(encoded_uncertainties) == 2) ? "var" : "cov"
    # return decoded or encoded?
    if transform_to_real
        decoded_outputs = decode_with_schedule(encoder_schedule, DataContainer(encoded_outputs), "out")
        
        decoded_covariances = zeros(output_dim, output_dim, size(encoded_uncertainties)[end])
        if var_or_cov == "var"
            for (i,col) in enumerate(eachcol(encoded_uncertainties))
               decoded_covariances[:,:,i] .= decode_with_schedule(encoder_schedule, Diagonal(col), "out")
            end
        else # == "cov"
            for (i,mat) in enumerate(eachslice(encoded_uncertainties, dims=3))
               decoded_covariances[:,:,i] .= decode_with_schedule(encoder_schedule, mat, "out")
            end
        end
            
        return get_data(decoded_outputs), eachslice(decoded_covariances,dims=3)
    else
        
        if encoded_output_dim > 1
            encoded_covariances_mat = zeros(output_dim, output_dim, size(encoded_uncertainties)[end])
            for (i,col) in enumerate(eachcol(encoded_uncertainties))
                encoded_covariances_mat[:,:,i] = Diagonal(col)
            end
            encoded_covariances = eachslice(encoded_covariances_mat,dims=3)
        else
            encoded_covariances = encoded_uncertainties
        end
        return encoded_outputs, encoded_covariances
    end
    
end


end
