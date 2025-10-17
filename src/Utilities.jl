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

using LowRankApprox
using TSVD

import LinearAlgebra: norm

export get_training_points
export create_compact_linear_map
export PairedDataContainerProcessor, DataContainerProcessor
export create_encoder_schedule,
    initialize_and_encode_with_schedule!,
    encode_with_schedule,
    decode_with_schedule,
    encode_data,
    encode_structure_matrix,
    decode_data,
    decode_structure_matrix,
    norm,
    norm_linear_map,
    isequal_linear,
    encoder_kwargs_from


const StructureMatrix = Union{UniformScaling, AbstractMatrix, AbstractVector, LinearMap} # The vector appears due to possible block-structured matrices (build=false)
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

"""
$(TYPEDSIGNATURES)

Extracts the relevant encoder kwargs from the observation as a NamedTuple. Contains,
- `:obs_noise_cov` as (unbuilt) noise covariance
- `:observation` as obs vector
"""
function encoder_kwargs_from(obs::OB) where {OB <: Observation}
    return (; obs_noise_cov = get_obs_noise_cov(obs, build = false), observation = get_obs(obs))
end

"""
$(TYPEDSIGNATURES)

Extracts the relevant encoder kwargs from the ObservationSeries as a NamedTuple. Assumes the same noise covariance for all observation vectors. Contains,
- `:obs_noise_cov` as (unbuilt) noise covariance of FIRST observation
- `:observation` as obs vector from all observations
"""
function encoder_kwargs_from(os::OS) where {OS <: ObservationSeries}
    observations = get_observations(os)
    obs_vec = [get_obs(obs) for obs in observations]
    obs_noise_cov = get_obs_noise_cov(observations[1], build = false)
    if !all([get_obs_noise_cov(observations[i], build = false) == obs_noise_cov for i in length(observations)])
        @warn("""
 Detected that observation covariances vary for different observations.
 Encoder kwarg `:obs_noise_cov` will be set to the FIRST of these covariances for the purpose of data processing.
 """)
    end
    return (; obs_noise_cov = obs_noise_cov, observation = obs_vec)
end

"""
$(TYPEDSIGNATURES)

Extracts the relevant encoder kwargs from the ParameterDistribution prior. Contains,
- `:prior_cov` as prior covariance
"""
function encoder_kwargs_from(prior::PD) where {PD <: ParameterDistribution}
    return (; prior_cov = cov(prior))
end

##  multiplication with observation covariance objects without building
"""
$(TYPEDSIGNATURES)

Produces a linear map of type `LinearMap` that can evaluates the stacked actions of the structure matrix in compact form. by calling say `linear_map.f(x)` or `linear_map.fc(x)` to obtain `Ax`, or `A'x`. This particular type can be used by packages like `TSVD.jl` or `IterativeSolvers.jl` for further computations.

This compact map constructs the following form of the Linear map f:

1. get compact form svd-plus-d form "USVt + D" of the `blocks`
2. create the f via stacking `A.U * A.S * A.Vt * xblock + A.D * xblock for (A,xblock) in  (As, x)`

kwargs:
------
When computing the svd internally from an abstract matrix
- `svd_dim_max=3000`: this switches to an approximate svd approach when applying to covariance matrices above dimension 3000
- `psvd_or_tsvd="psvd"`: use psvd or tsvd for approximating svd for large matrices
- `tsvd_max_rank=50`: when using tsvd, what max rank to use. high rank = higher accuracy
- `psvd_kwargs=(; rtol=1e-2)`: when using psvd, what kwargs to pass. lower rtol = higher accuracy

Recommended: quick & inaccurate -> slow and more accurate
- very large matrices - start with tsvd with very low rank, and increase
- mid-size matrices - psvd with very high rtol, and decrease
"""
function create_compact_linear_map(
    A;
    svd_dim_max = 3000,
    psvd_or_tsvd = "psvd",
    tsvd_max_rank = 50,
    psvd_kwargs = (; rtol = 1e-1),
)
    Avec = isa(A, AbstractVector) ? A : [A]

    # explicitly write the loop here:
    Us = []
    Ss = []
    VTs = []
    ds = []
    batches = []
    shift = 0
    for a in Avec
        bsize = 0
        if isa(a, UniformScaling)
            throw(
                ArgumentError(
                    "Detected `UniformScaling` (i.e. \"λI\") StructureMatrix, and unable to infer dimensionality. \n Please recast this as a diagonal matrix, defining \"λI(d)\" for dimension d",
                ),
            )
        end
        if isa(a, AbstractMatrix)
            if size(a, 1) <= svd_dim_max
                svda = svd(a)
                push!(Us, svda.U)
                push!(Ss, svda.S)
                push!(VTs, svda.Vt)
                push!(ds, zeros(size(a, 1)))
                bsize = size(a, 1)
            else
                if psvd_or_tsvd == "psvd"
                    svda = psvd(a; psvd_kwargs...)
                    push!(Us, svda[1])
                    push!(Ss, svda[2])
                    push!(VTs, svda[3]')
                else
                    svda = tsvd_mat(a, min(tsvd_max_rank, size(a, 1) - 1))
                    push!(Us, svda.U)
                    push!(Ss, svda.S)
                    push!(VTs, svda.Vt)
                end
                push!(ds, zeros(size(a, 1)))
                bsize = size(a, 1)
            end
        elseif isa(a, SVD)
            svda = a
            push!(Us, svda.U)
            push!(Ss, svda.S)
            push!(VTs, svda.Vt)
            push!(ds, zeros(size(a.U, 1)))
            bsize = size(a.U, 1)
        elseif isa(a, SVDplusD)
            svda = a.svd_cov
            diaga = (a.diag_cov).diag
            push!(Us, svda.U)
            push!(Ss, svda.S)
            push!(VTs, svda.Vt)
            push!(ds, diaga)
            bsize = length(diaga)
        end

        batch = (shift + 1):(shift + bsize)
        push!(batches, batch)
        shift = batch[end]
    end

    # then create the LinearMap with entries  (f(x) = A*x, f(x) = A'*x, size(A,1), size(A,2))
    # LinearMaps can only be applied to vectors in general, so we only provide this argumentation

    Amap = LinearMap(
        x -> reduce(
            vcat,
            [U * (S .* (Vt * x[batch])) + d .* x[batch] for (U, S, Vt, d, batch) in zip(Us, Ss, VTs, ds, batches)],
        ),
        x -> reduce(
            vcat,
            [Vt' * (S .* (U' * x[batch])) + d .* x[batch] for (U, S, Vt, d, batch) in zip(Us, Ss, VTs, ds, batches)],
        ),
        sum(size(U, 1) for U in Us),
        sum(size(Vt, 2) for Vt in VTs),
    )

    return Amap

end

"""
$(TYPEDSIGNATURES)

Approximately computes the norm of a `LinearMap` object. For `Amap` associated with matrix `A`, `norm_linear_map(Amap,p)≈norm(A,p)`. Can be aliased as `norm()`

kwargs
------
- n_eval(=nothing): number of mat-vec products to apply in the approximation (larger is more accurate). default performs `size(map,2)` products
- rng(=Random.default_rng()): random number generator

"""
function norm_linear_map(A::LM, p::Real = 2; n_eval = nothing, rng = Random.default_rng()) where {LM <: LinearMap}
    m, n = size(A)

    # use unit-normalized gaussian vectors
    n_basis = isa(n_eval, Nothing) ? n : n_eval
    samples = randn(n, n_basis)
    for i in 1:size(samples, 2)
        samples[:, i] /= norm(samples[:, i])
    end
    out = zeros(m, n_basis)
    mul!(out, A, samples)# must use mul! for multiply with lin map to return a matrix)
    norm_val = (n / n_basis)^(1 / p) * norm(out, p)

    return norm_val
end

LinearAlgebra.norm(A::LM, p::Real = 2; lm_kwargs...) where {LM <: LinearMap} = norm_linear_map(A, p; lm_kwargs...)

# Data processing tooling:

abstract type DataProcessor end
abstract type PairedDataContainerProcessor <: DataProcessor end # tools that operate on inputs and outputs 
abstract type DataContainerProcessor <: DataProcessor end # tools that operate on only inputs or outputs
# define how to have equality
# this gets messy with LinearMaps, 

"""
$(TYPEDSIGNATURES)

Tests equality for a LinearMap on a standard basis of the input space. Note that this operation requires a matrix multiply per input dimension so can be expensive.

Kwargs:
-------
- n_eval (=nothing): the number of basis vectors to compare against (randomly selected without replacement if `n_eval < size(A,1)`)
- tol (=2*eps()): the tolerance for equality on evaluation per entry
- rng (=default_rng()): When provided, and `n_eval < size(A,1)`; a random subset of the basis is compared, using this `rng`.
- up_to_sign(=false): Only assess equality up to a sign-error (sufficient for e.g. encoder/decoder matrices)
"""
function isequal_linear(
    A::LM1,
    B::LM2;
    tol = 2 * eps(),
    n_eval = nothing,
    rng = Random.default_rng(),
    up_to_sign = false,
) where {LM1 <: LinearMap, LM2 <: LinearMap}
    m, n = size(A)
    if !(n == size(B, 2))
        @warn "Comparing equality of linear maps with size ($(m), $(n)) and ($(size(B,1)), $(size(B,2))). Was this intended?"
        return false
    end
    if !(m == size(B, 1))
        @warn "Comparing equality of linear maps with size ($(m), $(n)) and ($(size(B,1)),$(size(B,2))). Was this intended?"
        return false
    end

    # test on standard basis (up to n_eval tests)
    basis_id = isa(n_eval, Nothing) ? collect(1:n) : randperm(rng, n)[1:n_eval]

    e = vec(zeros(eltype(A), n))
    for j in basis_id
        e[j] += 1
        if up_to_sign
            AmB = abs.(A * e) - abs.(B * e)
        else
            AmB = A * e - B * e
        end
        if !(norm(AmB) <= sqrt(n) * tol)
            return false
        end
        e[j] -= 1
    end
    return true
end

function Base.:(==)(a::LM1, b::LM2) where {LM1 <: LinearMap, LM2 <: LinearMap}
    n = size(a, 2)
    if n < 1e4 # gets expensive
        return isequal_linear(a, b; tol = 1e-12)
    else
        return isequal_linear(a, b; n_eval = Int(floor(sqrt(n))), tol = 1e-12) # 1e4 compares ~ 100 evals, 1e7 compares ~ 3000 evals
    end
end

Base.:(==)(a::DCP1, b::DCP2) where {DCP1 <: DataContainerProcessor, DCP2 <: DataContainerProcessor} =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(DCP1))

Base.:(==)(a::PDCP1, b::PDCP2) where {PDCP1 <: PairedDataContainerProcessor, PDCP2 <: PairedDataContainerProcessor} =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(PDCP1))
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
    samples_in::Union{Nothing, StructureVector} = nothing,
    samples_out::Union{Nothing, StructureVector} = nothing,
) where {VV <: AbstractVector, PDC <: PairedDataContainer}
    processed_io_pairs = deepcopy(io_pairs)

    # We additionally convert the `mats` into a linear-maps for flexible handling of massive covariances. In a matrix-free manner
    input_structure_mats = deepcopy(input_structure_mats)
    if !isnothing(prior_cov)
        (input_structure_mats[:prior_cov] = prior_cov)
    end
    for (key, val) in input_structure_mats
        if isa(val, UniformScaling) # remove this annoying case immediately
            input_dim = size(get_inputs(io_pairs), 1)
            input_structure_mats[key] = create_compact_linear_map(val(input_dim))
        else
            input_structure_mats[key] = create_compact_linear_map(val)
        end
    end

    output_structure_mats = deepcopy(output_structure_mats)
    if !isnothing(obs_noise_cov)
        (output_structure_mats[:obs_noise_cov] = obs_noise_cov)
    end
    for (key, val) in output_structure_mats
        if isa(val, UniformScaling) # remove this annoying case immediately
            output_dim = size(get_outputs(io_pairs), 1)
            output_structure_mats[key] = create_compact_linear_map(val(output_dim))
        else
            output_structure_mats[key] = create_compact_linear_map(val)
        end
    end

    input_structure_vecs = deepcopy(input_structure_vecs)
    if !isnothing(samples_in)
        (input_structure_vecs[:samples_in] = samples_in)
    end

    output_structure_vecs = deepcopy(output_structure_vecs)
    if !isnothing(observation)
        (output_structure_vecs[:observation] = observation)
    end
    if !isnothing(samples_out)
        (output_structure_vecs[:samples_out] = samples_out)
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
) where {VV <: AbstractVector, SM1 <: StructureMatrix, SM2 <: StructureMatrix, PDC <: PairedDataContainer}
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
include("Utilities/likelihood_informed.jl")

end # module
