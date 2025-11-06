module Utilities

using DocStringExtensions
using LinearAlgebra
using LinearMaps
using Statistics
using StatsBase
using Random
using ..EnsembleKalmanProcesses
const EKP = EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using ..DataContainers

export get_training_points

export PairedDataContainerProcessor, DataContainerProcessor
export create_encoder_schedule,
    initialize_and_encode_with_schedule!,
    encode_with_schedule,
    decode_with_schedule,
    encode_data,
    encode_structure_matrix,
    decode_data,
    decode_structure_matrix


const StructureMatrix = Union{UniformScaling, AbstractMatrix, AbstractVector} # The vector appears due to possible block-structured matrices (build=false)
const StructureVector = Union{AbstractVector, AbstractMatrix} # In case of a matrix, the columns should be seen as vectors

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the training points needed to train the Gaussian process regression.

- `ekp` - EnsembleKalmanProcess holding the parameters and the data that were produced
  during the Ensemble Kalman (EK) process.
- `train_iterations` - Number (or indices) EK layers/iterations to train on.

"""
function get_training_points(
    ekp::EKP.EnsembleKalmanProcess{FT, IT, P},
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

# Using Observation Objects:

function encoder_kwargs_from(obs::OB) where {OB <: Observation}
    return (obs_noise_cov = get_covs(obs, build=false), observation = get_obs(obs))
end

function encoder_kwargs_from(os::OS) where {OS <: ObservationSeries}
    obs_vec = get_observations(os)
    return [encoder_kwargs_from(obs) for obs in obs_vec]
end

function encoder_kwargs_from(prior::PD; rng=Random.default_rng(), n_samples = 100) where {PD <: ParameterDistribution}
    return (prior_cov = cov(prior), prior_samples_in = sample(rng, prior, n_samples))
end

##  multiplication with observation covariance objects without building
"""
$(TYPEDSIGNATURES)

Left-multiply `X` by structure matrix `A` without building it (if provided in a compact form).

This is useful when A is high dimensional and provided as an `SVD` or `SumOfCovariances` etc. object from EnsembleKalmanProcesses. 
"""
function lmul_compact(A, X::AVorM) where {AVorM <: AbstractVecOrMat}
    # A is presumed a vector, of (compact) matrix types.
    return isa(A, AbstractVector) ? EKP.lmul_without_build(A,X) : EKP.lmul_without_build([A],X)
end

"""
$(TYPEDSIGNATURES)

Produces a linear map of type `LinearMap` that can evaluates the stacked actions of the structure matrix in compact form. by calling say `linear_map.f(x)` or `linear_map.fc(x)` to obtain `Ax`, or `A'x`. This particular type can be used by packages like `TSVD.jl` or `IterativeSolvers.jl` for further computations.

This compact map constructs the following form of the Linear map f:

1. get compact form svd-plus-d form "USVt + D" of the `blocks`
2. create the f via stacking `A.U * A.S * A.Vt * x[block] + A.D * x[block] for (A,block) in blocks`

kwargs:
------
When computing the svd internally from an abstract matrix
- `svd_dim_max=4000`: this switches to an approximate tsvd approach when applying to covariance matrices above dimension 4000 
- `tsvd_rank=50`: when using tsvd, what rank to truncate at.

"""
function create_compact_linear_map(A; svd_dim_max = 4000, tsvd_rank=50)
    Avec = isa(A, AbstractVector) ? A  : [A]

    # explicitly write the loop here:
    Us = []
    Ss = []
    VTs = []
    ds = []
    batches=[]
    shift = 0
    for a in Avec
        if isa(a, AbstractMatrix)
            if size(a,1) <= svd_dim_max
                svda = svd(a)
                diaga = zeros(size(a,1))
                bsize = size(a,1)                    
            else
                svda = EKP.tsvd_mat(a, min(tsvd_rank, size(a,1)-1)) # swap to tsvd for performance
                diaga = zeros(size(a,1))
                bsize = size(a,1)
            end                
        elseif isa(a, SVD)
            svda = a
            diaga = zeros(size(a.U,1))
            bsize = size(a.U,1)
        elseif isa(a, SVDplusD)  
            svda = a.svd_cov
            diaga = (a.diag_cov).diag
            bsize = length(diaga)
        end
        push!(Us, svda.U)
        push!(Ss, svda.S)
        push!(VTs, svda.Vt)
        push!(ds, diaga)
        
        batch = shift+1:shift+bsize
        push!(batches, batch)
        shift = batch[end]
    end
        
    # then create the LinearMap with entries  f(x) = A*x, f(x) = A'*x, size(A,1), size(A,2)
    # LinearMaps can only be applied to vectors in general, so we only provide this argumentation
    
    Amap = LinearMap(
    x -> reduce(vcat, [U * (S .* (Vt * x[batch])) + d .* x[batch] for (U,S,Vt,d, batch) in zip(Us,Ss,VTs,ds,batches)]), 
    x -> reduce(vcat, [Vt' * (S .* (U' * x[batch])) + d .* x[batch] for (U,S,Vt,d, batch) in zip(Us,Ss,VTs,ds,batches)]),        
        sum(size(U, 1) for U in Us),
        sum(size(Vt, 2) for Vt in VTs),
    )

    return Amap
    
end


#=
"""
$(TYPEDSIGNATURES)

svd of a sum of two matrices `A` and `B` in svd form without building A+B directly. 

This is useful when A and B are low rank, but high dimensional and provided as SVD types. (e.g. from `EKP.tsvd_cov_from_samples`)
"""
function svd_of_sum(svdA::SS1,svdB::SS2) where { SS1 <: SVD, SS2 <: SVD}
    
    # treat A + B as just "low rank"
    # can write A+B = L * R = [Ua Ub] * [ Sa * Va'; Sb * Vb']
    L = [svdA.U svdB.U] # N x (r1+r2)
    R = [Diagonal(svdA.S) * svdA.Vt ; Diagonal(svdB.S) * svdB.Vt] # (r1+r2) x N
    qrl = qr(L) # Q1 R1
    qrr = qr(R') # Q2 R2
    # A+B = Q_1 * R_1 * R_2' * Q_2' = Q1 * (ss.U * ss.S * ss.V' * Q2' = (Q1 ss.U) * ss.S * (Q2 ss.V)'
    ss = svd(qrl.R*qrr.R') 

    return SVD(qrl.Q*ss.U, ss.S, ss.Vt*qrr.Q') # build the new svd
    
end
        
function inv_sqrt_of_svdplusd(a)
    D = a.diag_cov
    D += sqrt(eps())*I # minimum tolerance for some operations below
    iD = inv(D) # 
    irD = sqrt.(iD)
    ss = a.svd_cov
    U = ss.U
    S = Diagonal(ss.S)
    
    # begin computing the whitening transform
    M = inv(S) + U'*iD*U
    riM = sqrt(inv(M))
    B = irD*U
    C = riM*B'*B*riM 
    ev = eigen(C)
    V = ev.vectors
    # if A = B*riM => 
    #    (I-AA')^{1/2} = I - A * Chat * A'
    #                  = I - A * V * Diag(chat) * V' * A'
    # LHS evals sqrt(1-σ_i^2), rhs evals 1 - chat * σ_i^2
    #     => chat = (1 - sqrt(1-σ_i^2))/(σ_i^2) # where σ_i is eval of A = sqrt(ev.values)
    ev_correction = Diagonal((1.0 .-sqrt.(1.0 .- ev.values)) ./ ev.values)
    
    # Then,  (I - BM^{-1}B')^{1/2} = I - B M^{-1/2} V sv V' M^{-1/2} B'
    rt_ImBiMB = I - B * riM * V * ev_correction * V' * riM * B'
    # Idea is that woodbury: Σ^{-1} = D^{-1/2} * ImBiMB * D^{-1/2}
    # Then W = rt_ImBiMB * D^{-1/2} satisfies W Σ W' = I 
    W = rt_ImBiMB * irD # perhaps keep this in a compact form until use:
    return W
end
=#



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

function get_structure_vec(structure_vecs, name = nothing)
    if isnothing(name)
        if length(structure_vecs) == 1
            return only(values(structure_vecs))
        elseif isempty(structure_vecs)
            throw(ArgumentError("Please provide a structure vector."))
        else
            throw(
                ArgumentError(
                    "Structure vectors $(collect(keys(structure_vecs))) are present. I received argument `name = nothing`, so I don't know which to use.",
                ),
            )
        end
    else
        if haskey(structure_vecs, name)
            return structure_vecs[name]
        else
            throw(ArgumentError("Structure vector $name not found. Options: $(collect(keys(structure_vecs)))."))
        end
    end
end

function get_structure_mat(structure_mats, name = nothing)
    if isnothing(name)
        if length(structure_mats) == 1
            return only(values(structure_mats))
        elseif isempty(structure_mats)
            throw(ArgumentError("Please provide a structure matrix."))
        else
            throw(
                ArgumentError(
                    "Structure matrices $(collect(keys(structure_mats))) are present. I received argument `name = nothing`, so I don't know which to use.",
                ),
            )
        end
    else
        if haskey(structure_mats, name)
            return structure_mats[name]
        else
            throw(ArgumentError("Structure matrix $name not found. Options: $(collect(keys(structure_mats)))."))
        end
    end
end

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
    proc::PairedDataContainerProcessor,
    data,
    structure_mats,
    structure_vecs,
    apply_to::AS,
) where {AS <: AbstractString}
    initialize_processor!(proc, get_data(data)..., structure_mats..., structure_vecs..., apply_to)
    return _encode_data(proc, data, apply_to)
end

function _initialize_and_encode_data!(
    proc::DataContainerProcessor,
    data,
    structure_mats,
    structure_vecs,
    apply_to::AS,
) where {AS <: AbstractString}
    input_data, output_data = get_data(data)
    input_structure_mats, output_structure_mats = structure_mats
    input_structure_vecs, output_structure_vecs = structure_vecs

    if apply_to == "in"
        initialize_processor!(proc, input_data, input_structure_mats, input_structure_vecs)
    elseif apply_to == "out"
        initialize_processor!(proc, output_data, output_structure_mats, output_structure_vecs)
    else
        bad_apply_to(apply_to)
    end

    return _encode_data(proc, data, apply_to)
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
This function creates the encoder scheduler that is also machine readable. E.g.,
```julia
enc_schedule = [
    (DataProcessor1(...), "in"), 
    (DataProcessor2(...), "out"), 
    (DataProcessor2(...), "out"),
    (PairedDataProcessor3(...),"in"), 
    (DataProcessor4(...), "in"),
    (DataProcessor4(...), "out"), 
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
    io_pairs::PDC;
    input_structure_mats = Dict{Symbol, StructureMatrix}(),
    output_structure_mats = Dict{Symbol, StructureMatrix}(),
    input_structure_vecs = Dict{Symbol, StructureVector}(),
    output_structure_vecs = Dict{Symbol, StructureVector}(),
    prior_cov::Union{Nothing, StructureMatrix} = nothing,
    obs_noise_cov::Union{Nothing, StructureMatrix} = nothing,
    observation::Union{Nothing, StructureVector} = nothing,
    prior_samples_in::Union{Nothing, StructureVector} = nothing,
    prior_samples_out::Union{Nothing, StructureVector} = nothing,
) where {VV <: AbstractVector, PDC <: PairedDataContainer}
    processed_io_pairs = deepcopy(io_pairs)

    input_structure_mats = deepcopy(input_structure_mats)
    if !isnothing(prior_cov)
        (input_structure_mats[:prior_cov] = prior_cov)
    end

    output_structure_mats = deepcopy(output_structure_mats)
    if !isnothing(obs_noise_cov)
        (output_structure_mats[:obs_noise_cov] = obs_noise_cov)
    end

    input_structure_vecs = deepcopy(input_structure_vecs)
    if !isnothing(prior_samples_in)
        (input_structure_vecs[:prior_samples_in] = prior_samples_in)
    end

    output_structure_vecs = deepcopy(output_structure_vecs)
    if !isnothing(observation)
        (output_structure_vecs[:observation] = observation)
    end
    if !isnothing(prior_samples_out)
        (output_structure_vecs[:prior_samples_out] = prior_samples_out)
    end

    # apply_to is the string "in", "out" etc.
    for (processor, apply_to) in encoder_schedule
        @info "Initialize encoding of data: \"$(apply_to)\" with $(processor)"

        processed = _initialize_and_encode_data!(
            processor,
            processed_io_pairs,
            (input_structure_mats, output_structure_mats),
            (input_structure_vecs, output_structure_vecs),
            apply_to,
        )

        if apply_to == "in"
            input_structure_mats = Dict{Symbol, StructureMatrix}(
                name => encode_structure_matrix(processor, mat) for (name, mat) in input_structure_mats
            )
            input_structure_vecs = Dict{Symbol, StructureVector}(
                name => encode_data(processor, vec) for (name, vec) in input_structure_vecs
            )
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            output_structure_mats = Dict{Symbol, StructureMatrix}(
                name => encode_structure_matrix(processor, mat) for (name, mat) in output_structure_mats
            )
            output_structure_vecs = Dict{Symbol, StructureVector}(
                name => encode_data(processor, vec) for (name, vec) in output_structure_vecs
            )
            processed_io_pairs = PairedDataContainer(get_inputs(processed_io_pairs), processed)
        end
    end

    return processed_io_pairs, input_structure_mats, output_structure_mats, input_structure_vecs, output_structure_vecs
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
    structure_matrix::SM,
    in_or_out::AS,
) where {VV <: AbstractVector, SM <: StructureMatrix, AS <: AbstractString}
    if in_or_out ∉ ["in", "out"]
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
    input_structure_mat::SM1,
    output_structure_mat::SM2,
) where {
    VV <: AbstractVector,
    SM1 <: StructureMatrix,
    SM2 <: StructureMatrix,
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
    structure_matrix::SM,
    in_or_out::AS,
) where {VV <: AbstractVector, SM <: StructureMatrix, AS <: AbstractString}
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
