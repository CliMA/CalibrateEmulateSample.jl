module Utilities



using DocStringExtensions
using LinearAlgebra
using Statistics
using StatsBase
using Random
using ..EnsembleKalmanProcesses
EnsembleKalmanProcess = EnsembleKalmanProcesses.EnsembleKalmanProcess
using ..DataContainers

export get_training_points

export PairedDataContainerProcessor, DataContainerProcessor
export create_encoder_schedule, initialize_and_encode_with_schedule!, encode_with_schedule, decode_with_schedule



"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the training points needed to train the Gaussian process regression.

- `ekp` - EnsembleKalmanProcess holding the parameters and the data that were produced
  during the Ensemble Kalman (EK) process.
- `train_iterations` - Number (or indices) EK layers/iterations to train on.

"""
function get_training_points(
    ekp::EnsembleKalmanProcess{FT, IT, P},
    train_iterations::Union{IT, AbstractVector{IT}},
) where {FT, IT, P}

    if !isa(train_iterations, AbstractVector)
        # Note u[end] does not have an equivalent g
        iter_range = (get_N_iterations(ekp) - train_iterations + 1):get_N_iterations(ekp)
    else
        iter_range = train_iterations
    end

    u_tp = []
    g_tp = []
    for i in iter_range
        push!(u_tp, get_u(ekp, i)) #N_parameters x N_ens
        push!(g_tp, get_g(ekp, i)) #N_data x N_ens
    end
    u_tp = hcat(u_tp...) # N_parameters x (N_ek_it x N_ensemble)]
    g_tp = hcat(g_tp...) # N_data x (N_ek_it x N_ensemble)

    training_points = PairedDataContainer(u_tp, g_tp, data_are_columns = true)

    return training_points
end


# Data processing tooling:

abstract type DataProcessor end
abstract type PairedDataContainerProcessor <: DataProcessor end # tools that operate on inputs and outputs 
abstract type DataContainerProcessor <: DataProcessor end # tools that operate on only inputs or outputs

# define how to have equality
Base.:(==)(a::DCP, b::DCP) where {DCP <: DataContainerProcessor} =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(DCP))
Base.:(==)(a::PDCP, b::PDCP) where {PDCP <: PairedDataContainerProcessor} =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(PDCP))


####


function _encode_data(proc::P, data, apply_to::AS) where {P <: DataProcessor, AS <: AbstractString}
    input_data, output_data = get_data(data)
    if apply_to == "in"
        return encode_data(proc, input_data)
    elseif apply_to == "out"
        return encode_data(proc, output_data)
    else
        bad_apply_to(apply_to)
    end
end

function _decode_data(proc::P, data, apply_to::AS) where {P <: DataProcessor, AS <: AbstractString}
    input_data, output_data = get_data(data)

    if apply_to == "in"
        return decode_data(proc, input_data)
    elseif apply_to == "out"
        return decode_data(proc, output_data)
    else
        bad_apply_to(apply_to)
    end
end

function _initialize_and_encode_data!(
    proc::P,
    data,
    structure_mats,
    apply_to::AS,
) where {P <: DataProcessor, AS <: AbstractString}
    if proc isa PairedDataContainerProcessor
        initialize_processor!(proc, data..., structure_mats..., apply_to)
    else
        input_data, output_data = get_data(data)
        input_structure_mat, output_structure_mat = structure_mats

        if apply_to == "in"
            initialize_processor!(proc, input_data, input_structure_mat)
        elseif apply_to == "out"
            initialize_processor!(proc, output_data, output_structure_mat)
        else
            bad_apply_to(apply_to)
        end
    end

    _encode_data(proc, data, apply_to)
end

"""
$TYPEDSIGNATURES

Create a flatter encoder schedule for the 
from the user's proposed schedule of the form:
```julia
enc_schedule = [
    (DataProcessor1(...), "in"), 
    (DataProcessor2(...), "out"), 
    (PairedDataProcessor3(...), "in"), 
    (DataProcessor4(...), "in_and_out"), 
]
```
This function creates the encoder scheduler that is also machine readable
```julia
enc_schedule = [
    (DataProcessor1(...), x -> get_inputs(x), "in"), 
    (DataProcessor2(...), x -> get_outputs(x), "out"), 
    (DataProcessor2(...), x -> get_outputs(x), "out"),
    (PairedDataProcessor3(...), x -> (get_outputs(x), get_outputs(x)), "in"), 
    (DataProcessor4(...), x -> get_inputs(x), "in"),
    (DataProcessor4(...), x -> get_outputs(x), "out"), 
]
```
and the decoder schedule is a copy of the encoder schedule reversed (and processors copied)
"""
function create_encoder_schedule(schedule_in::VV) where {VV <: AbstractVector}
    encoder_schedule = []
    for (processor, apply_to) in schedule_in
        # converts the string into the extraction of data
        if apply_to ∈ ["in", "out"]
            push!(encoder_schedule, (processor, apply_to))
        elseif apply_to == "in_and_out"
            push!(encoder_schedule, (processor, "in"))
            push!(encoder_schedule, (deepcopy(processor), "out"))
        else
            @warn(
                "Expected schedule keywords ∈ {\"in\",\"out\",\"in_and_out\"}. Received $(apply_to), ignoring processor $(processor)..."
            )
        end
    end

    return encoder_schedule
end

"""
Create size-1 encoder schedule with a tuple of `(DataProcessor1(...), apply_to)` with `apply_to = "in"`, `"out"` or `"in_and_out"`.
"""
create_encoder_schedule(schedule_in::TT) where {TT <: Tuple} = create_encoder_schedule([schedule_in])

# Functions to encode/decode with uninitialized schedule (require structure matrices as input)
"""
$TYPEDSIGNATURES

Takes in the created encoder schedule (See [`create_encoder_schedule`](@ref)), and initializes it, and encodes the paired data container, and structure matrices with it.
"""
function initialize_and_encode_with_schedule!(
    encoder_schedule::VV,
    io_pairs::PDC,
    input_structure_mat::USorM1,
    output_structure_mat::USorM2,
) where {
    VV <: AbstractVector,
    PDC <: PairedDataContainer,
    USorM1 <: Union{UniformScaling, AbstractMatrix},
    USorM2 <: Union{UniformScaling, AbstractMatrix},
}
    processed_io_pairs = deepcopy(io_pairs)
    processed_input_structure_mat = deepcopy(input_structure_mat)
    processed_output_structure_mat = deepcopy(output_structure_mat)

    # apply_to is the string "in", "out" etc.
    for (processor, apply_to) in encoder_schedule
        @info "Initialize encoding of data: \"$(apply_to)\" with $(processor)"

        processed = _initialize_and_encode_data!(
            processor,
            processed_io_pairs,
            (processed_input_structure_mat, processed_output_structure_mat),
            apply_to,
        )

        if apply_to == "in"
            processed_input_structure_mat = encode_structure_matrix(processor, processed_input_structure_mat)
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            processed_output_structure_mat = encode_structure_matrix(processor, processed_output_structure_mat)
            processed_io_pairs = PairedDataContainer(get_inputs(processed_io_pairs), processed)
        end
    end

    return processed_io_pairs, processed_input_structure_mat, processed_output_structure_mat
end


# Functions to encode/decode with initialized schedule
"""
$TYPEDSIGNATURES

Takes in an already initialized encoder schedule, and encodes a `DataContainer`, the `in_or_out` string indicates if the data is input `"in"` or output `"out"` data (and thus encoded differently)
"""
function encode_with_schedule(
    encoder_schedule::VV,
    data_container::DC,
    in_or_out::AS,
) where {VV <: AbstractVector, DC <: DataContainer, AS <: AbstractString}
    if in_or_out ∉ ["in", "out"]
        bad_in_or_out(in_or_out)
    end
    processed_container = deepcopy(data_container)

    # apply_to is the string "in", "out" etc.
    for (processor, apply_to) in encoder_schedule
        if apply_to == in_or_out
            processed = encode_data(processor, get_data(processed_container))
            processed_container = DataContainer(processed)
        end
    end

    return processed_container
end

"""
$TYPEDSIGNATURES

Takes in an already initialized encoder schedule, and encodes a structure matrix, the `in_or_out` string indicates if the structure matrix is for input `"in"` or output `"out"` space (and thus encoded differently)
"""
function encode_with_schedule(
    encoder_schedule::VV,
    structure_matrix::USorM,
    in_or_out::AS,
) where {VV <: AbstractVector, USorM <: Union{UniformScaling, AbstractMatrix}, AS <: AbstractString}
    if !(in_or_out ∈ ["in", "out"])
        bad_in_or_out(in_or_out)
    end
    processed_structure_matrix = deepcopy(structure_matrix)

    # apply_to is the string "in", "out" etc.
    for (processor, apply_to) in encoder_schedule
        if apply_to == in_or_out
            processed_structure_matrix = encode_structure_matrix(processor, processed_structure_matrix)
        end
    end

    return processed_structure_matrix
end

"""
$TYPEDSIGNATURES

Takes in an already initialized encoder schedule, and decodes a `DataContainer`, and structure matrices with it, the `in_or_out` string indicates if the data is input `"in"` or output `"out"` data (and thus decoded differently)
"""
function decode_with_schedule(
    encoder_schedule::VV,
    io_pairs::PDC,
    input_structure_mat::USorM1,
    output_structure_mat::USorM2,
) where {
    VV <: AbstractVector,
    USorM1 <: Union{UniformScaling, AbstractMatrix},
    USorM2 <: Union{UniformScaling, AbstractMatrix},
    PDC <: PairedDataContainer,
}
    processed_io_pairs = deepcopy(io_pairs)
    processed_input_structure_mat = deepcopy(input_structure_mat)
    processed_output_structure_mat = deepcopy(output_structure_mat)

    # apply_to is the string "in", "out" etc.
    for idx in reverse(eachindex(encoder_schedule))
        (processor, apply_to) = encoder_schedule[idx]
        processed = _decode_data(processor, processed_io_pairs, apply_to)

        if apply_to == "in"
            processed_input_structure_mat = decode_structure_matrix(processor, processed_input_structure_mat)
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            processed_output_structure_mat = decode_structure_matrix(processor, processed_output_structure_mat)
            processed_io_pairs = PairedDataContainer(get_inputs(processed_io_pairs), processed)
        else
            bad_apply_to(apply_to)
        end
    end

    return processed_io_pairs, processed_input_structure_mat, processed_output_structure_mat
end

"""
$TYPEDSIGNATURES

Takes in an already initialized encoder schedule, and decodes a `DataContainer`, the `in_or_out` string indicates if the data is input `"in"` or output `"out"` data (and thus decoded differently)
"""
function decode_with_schedule(
    encoder_schedule::VV,
    data_container::DC,
    in_or_out::AS,
) where {VV <: AbstractVector, DC <: DataContainer, AS <: AbstractString}
    if in_or_out ∉ ["in", "out"]
        bad_in_or_out(in_or_out)
    end
    processed_container = deepcopy(data_container)

    # apply_to is the string "in", "out" etc.
    for idx in reverse(eachindex(encoder_schedule))
        (processor, apply_to) = encoder_schedule[idx]
        if apply_to == in_or_out
            processed = decode_data(processor, get_data(processed_container))
            processed_container = DataContainer(processed)
        end
    end

    return processed_container
end

"""
$TYPEDSIGNATURES

Takes in an already initialized encoder schedule, and decodes a structure matrix, the `in_or_out` string indicates if the structure matrix is for input `"in"` or output `"out"` space (and thus decoded differently)
"""
function decode_with_schedule(
    encoder_schedule::VV,
    structure_matrix::USorM,
    in_or_out::AS,
) where {VV <: AbstractVector, USorM <: Union{UniformScaling, AbstractMatrix}, AS <: AbstractString}
    if in_or_out ∉ ["in", "out"]
        bad_in_or_out(in_or_out)
    end
    processed_structure_matrix = deepcopy(structure_matrix)

    # apply_to is the string "in", "out" etc.
    for idx in reverse(eachindex(encoder_schedule))
        (processor, apply_to) = encoder_schedule[idx]
        if apply_to == in_or_out
            processed_structure_matrix = decode_structure_matrix(processor, processed_structure_matrix)
        end
    end

    return processed_structure_matrix
end


# Errors

function bad_apply_to(apply_to::AS) where {AS <: AbstractString}
    throw(
        ArgumentError(
            "processer can only be applied to inputs (\"in\") or outputs (\"out\"). received $(apply_to). \n Please use `create_encoder_schedule` prior to encoding/decoding to ensure correct schedule format",
        ),
    )
end

function bad_in_or_out(in_or_out::AS) where {AS <: AbstractString}
    throw(
        ArgumentError(
            "`in_or_out` must be either \"in\" (data is an input) or \"out\" (data is an output). Received $(in_or_out)",
        ),
    )
end


# Processors

include("Utilities/canonical_correlation.jl")
include("Utilities/decorrelator.jl")
include("Utilities/elementwise_scaler.jl")

end # module
