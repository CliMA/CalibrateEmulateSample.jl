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
export orig2zscore
export zscore2orig

export PairedDataContainerProcessor, DataContainerProcessor
export UnivariateAffineScaling, QuartileScaling, MinMaxScaling, ZScoreScaling, Standardizer

export quartile_scale, minmax_scale, zscore_scale, standardize
export get_type, get_shift, get_scale, get_data_mean, get_data_reduction_mat, get_data_inflation_mat, get_rank
export create_encoder_schedule, encode_with_schedule, decode_with_schedule
export initialize_processor!,
    initialize_and_encode_data!, encode_data, decode_data, encode_obs_noise_cov, decode_obs_noise_cov



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

function orig2zscore(X::AbstractVector{FT}, mean::AbstractVector{FT}, std::AbstractVector{FT}) where {FT}
    # Compute the z scores of a vector X using the given mean
    # and std
    Z = zeros(size(X))
    for i in 1:length(X)
        Z[i] = (X[i] - mean[i]) / std[i]
    end
    return Z
end

function orig2zscore(X::AbstractMatrix{FT}, mean::AbstractVector{FT}, std::AbstractVector{FT}) where {FT}
    # Compute the z scores of matrix X using the given mean and
    # std. Transformation is applied column-wise.
    Z = zeros(size(X))
    n_cols = size(X)[2]
    for i in 1:n_cols
        Z[:, i] = (X[:, i] .- mean[i]) ./ std[i]
    end
    return Z
end

function zscore2orig(Z::AbstractVector{FT}, mean::AbstractVector{FT}, std::AbstractVector{FT}) where {FT}
    # Transform X (a vector of z scores) back to the original
    # values
    X = zeros(size(Z))
    for i in 1:length(X)
        X[i] = Z[i] .* std[i] .+ mean[i]
    end
    return X
end

function zscore2orig(Z::AbstractMatrix{FT}, mean::AbstractVector{FT}, std::AbstractVector{FT}) where {FT}
    X = zeros(size(Z))
    # Transform X (a matrix of z scores) back to the original
    # values. Transformation is applied column-wise.
    n_cols = size(Z)[2]
    for i in 1:n_cols
        X[:, i] = Z[:, i] .* std[i] .+ mean[i]
    end
    return X
end

# Data processing tooling:

abstract type PairedDataContainerProcessor end # tools that operate on inputs and outputs 
abstract type DataContainerProcessor end # tools that operate on only inputs or outputs

abstract type UnivariateAffineScaling end
abstract type QuartileScaling <: UnivariateAffineScaling end
abstract type MinMaxScaling <: UnivariateAffineScaling end
abstract type ZScoreScaling <: UnivariateAffineScaling end

# Processors
"""
$(TYPEDEF)

The AffineScaler{T} will create an encoding of the data_container via affine transformations:
- quartile_scale() : creates `QuartileScaling`, encoding with the transform (x - Q2(x))/(Q3(x) - Q1(x))
- minmax_scale() : creates `MinMaxScaling`, encoding with the transform (x - min(x))/(max(x) - min(x))
- zscore_scale() : creates `ZScoreScaling`, encoding with the (univariate) transform (x-μ)/σ
"""
struct AffineScaler{T, VV <: AbstractVector} <: DataContainerProcessor
    "storage for the shift applied to data"
    shift::VV
    "storage for the scaling"
    scale::VV
end

quartile_scale() = AffineScaler(QuartileScaling)
quartile_scale(T::Type) = AffineScaler(QuartileScaling, T)
minmax_scale() = AffineScaler(MinMaxScaling)
minmax_scale(T::Type) = AffineScaler(MinMaxScaling, T)
zscore_scale() = AffineScaler(ZScoreScaling)
zscore_scale(T::Type) = AffineScaler(ZScoreScaling, T)

AffineScaler(::Type{UAS}) where {UAS <: UnivariateAffineScaling} =
    AffineScaler{UAS, Vector{Float64}}(Float64[], Float64[])
AffineScaler(::Type{UAS}, T::Type) where {UAS <: UnivariateAffineScaling} = AffineScaler{UAS, Vector{T}}(T[], T[])

get_type(as::AffineScaler{T, VV}) where {T, VV} = T
get_shift(as::AffineScaler) = as.shift
get_scale(as::AffineScaler) = as.scale

function Base.show(io::IO, as::AffineScaler)
    out = "AffineScaler: $(get_type(as))"
    print(io, out)
end

function initialize_processor!(
    as::AffineScaler,
    data::MM,
    T::Type{QS},
) where {MM <: AbstractMatrix, QS <: QuartileScaling}
    quartiles_vec = [quantile(dd, [0.25, 0.5, 0.75]) for dd in eachrow(data)]
    quartiles_mat = reduce(hcat, quartiles_vec) # 3 rows: Q1, Q2, and Q3
    append!(get_shift(as), quartiles_mat[2, :])
    append!(get_scale(as), (quartiles_mat[3, :] - quartiles_mat[1, :]))
end

function initialize_processor!(
    as::AffineScaler,
    data::MM,
    T::Type{MMS},
) where {MM <: AbstractMatrix, MMS <: MinMaxScaling}
    minmax_vec = [[minimum(dd), maximum(dd)] for dd in eachrow(data)]
    minmax_mat = reduce(hcat, minmax_vec) # 3 rows: Q1, Q2, and Q3
    append!(get_shift(as), minmax_mat[1, :])
    append!(get_scale(as), (minmax_mat[2, :] - minmax_mat[1, :]))
end

function initialize_processor!(
    as::AffineScaler,
    data::MM,
    T::Type{ZSS},
) where {MM <: AbstractMatrix, ZSS <: ZScoreScaling}
    stat_vec = [[mean(dd), std(dd)] for dd in eachrow(data)]
    stat_mat = reduce(hcat, stat_vec) # 3 rows: Q1, Q2, and Q3
    append!(get_shift(as), stat_mat[1, :])
    append!(get_scale(as), stat_mat[2, :])
end

function initialize_processor!(as::AffineScaler, data::MM) where {MM <: AbstractMatrix}
    if length(get_shift(as)) == 0
        T = get_type(as)
        initialize_processor!(as, data, T)
    end
end

function encode_data(as::AffineScaler, data::MM) where {MM <: AbstractMatrix}
    out = deepcopy(data)
    for i in 1:size(out, 1)
        out[i, :] .-= get_shift(as)[i]
        out[i, :] /= get_scale(as)[i]
    end
    return out
end

function decode_data(as::AffineScaler, data::MM) where {MM <: AbstractMatrix}
    out = deepcopy(data)
    for i in 1:size(out, 1)
        out[i, :] *= get_scale(as)[i]
        out[i, :] .+= get_shift(as)[i]
    end
    return out
end

initialize_processor!(as::AffineScaler, data::MM, structure_matrix) where {MM <: AbstractMatrix} =
    initialize_processor!(as, data)
encode_data(as::AffineScaler, data::MM, structure_matrix) where {MM <: AbstractMatrix} = encode_data(as, data)
decode_data(as::AffineScaler, data::MM, structure_matrix) where {MM <: AbstractMatrix} = decode_data(as, data)

function encode_structure_matrix(as::AffineScaler, structure_matrix::MM) where {MM <: AbstractMatrix}
    return Diagonal(1 ./ get_scale(as) .^ 2) * structure_matrix
end

function decode_structure_matrix(as::AffineScaler, enc_structure_matrix::MM) where {MM <: AbstractMatrix}
    return Diagonal(get_scale(as) .^ 2) * enc_structure_matrix
end



"""
$(TYPEDEF)

Standardizes the data to a multivariate N(0,I) distribution via `(xcov)^{-1/2}*(x-x_mean)`, along with rank reduction if data is low rank, or if the user provides a rank
"""
struct Standardizer{VV1 <: AbstractVector, VV2 <: AbstractVector, VV3 <: AbstractVector} <: DataContainerProcessor
    "user-provided rank for additional truncation of the space. used if rank < rank(data_cov)"
    rank::Int
    "storage for the data mean"
    data_mean::VV1
    "storage for wide-matrix representing `data_cov^{-1/2}`"
    data_reduction_mat::VV2
    "storage for tall-matrix representing `data_cov^{1/2}`"
    data_inflation_mat::VV3
end

standardize() = Standardizer(typemax(Int), Float64[], Any[], Any[])
standardize(T::Type) = Standardizer(typemax(Int), T[], Any[], Any[])
standardize(T1::Type, T2::Type, T3::Type) = Standardizer(typemax(Int), T1[], T2[], T3[])
standardize(rk::Int) = Standardizer(rk, Float64[], Any[], Any[])
standardize(rk::Int, T::Type) = Standardizer(rk, T[], Any[], Any[])
standardize(rk::Int, T1::Type, T2::Type, T3::Type) = Standardizer(rk, T1[], T2[], T3[])

get_data_mean(ss::Standardizer) = ss.data_mean
get_data_reduction_mat(ss::Standardizer) = ss.data_reduction_mat
get_data_inflation_mat(ss::Standardizer) = ss.data_inflation_mat
get_rank(ss::Standardizer) = ss.rank

function Base.show(io::IO, ss::Standardizer)
    if get_rank(ss) < typemax(Int)
        out = "Standardizer, user-rank=$(get_rank(ss))"
    else
        out = "Standardizer"
    end
    print(io, out)
end

function initialize_processor!(ss::Standardizer, data::MM) where {MM <: AbstractMatrix}
    if length(get_data_mean(ss)) == 0
        append!(get_data_mean(ss), mean(data, dims = 2))
    end
    if length(get_data_reduction_mat(ss)) == 0
        # can use tsvd here so we only compute up to rk
        data_cov = cov(data, dims = 2)
        svdc = svd(data_cov)
        rk = min(rank(data_cov), max(get_rank(ss), 1))
        # Want to use the nonsquare singular vector
        if size(svdc.U, 1) == size(svdc.U, 2)
            mat = svdc.Vt
        else
            mat = svdc.U'
        end

        reduction_mat = Diagonal(1 ./ sqrt.(svdc.S[1:rk])) * mat[1:rk, :]
        inflation_mat = mat[1:rk, :]' * Diagonal(sqrt.(svdc.S[1:rk]))
        push!(get_data_reduction_mat(ss), reduction_mat)
        push!(get_data_inflation_mat(ss), inflation_mat)
    end
end

function encode_data(ss::Standardizer, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(ss)
    reduction_mat = get_data_reduction_mat(ss)[1]
    return reduction_mat * (data .- data_mean)
end

function decode_data(ss::Standardizer, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(ss)
    inflation_mat = get_data_inflation_mat(ss)[1]
    return inflation_mat * data .+ data_mean
end

initialize_processor!(ss::Standardizer, data::MM, structure_matrix) where {MM <: AbstractMatrix} =
    initialize_processor!(ss, data)
encode_data(ss::Standardizer, data::MM, structure_matrix) where {MM <: AbstractMatrix} = encode_data(ss, data)
decode_data(ss::Standardizer, data::MM, structure_matrix) where {MM <: AbstractMatrix} = decode_data(ss, data)

function encode_structure_matrix(ss::Standardizer, structure_matrix::MM) where {MM <: AbstractMatrix}
    reduction_mat = get_data_reduction_mat(ss)[1]
    return reduction_mat * structure_matrix * reduction_mat'
end

function decode_structure_matrix(ss::Standardizer, enc_structure_matrix::MM) where {MM <: AbstractMatrix}
    inflation_mat = get_data_inflation_mat(ss)[1]
    return inflation_mat * enc_structure_matrix * inflation_mat'
end


"""
$(TYPEDEF)

Decorrelate the data via an SVD decomposition using a structure_matrix (`prior_cov` for inputs, `obs_noise_cov` for outputs), with optional truncation of singular vectors corresponding to largest singular values.

# Fields
$(TYPEDFIELDS)
"""
struct Decorrelater{MM1, MM2, FT} <: DataContainerProcessor
    "the structure matrix - provided by the user"
    structure_matrix::MM1
    "the inverse of the structure matrix"
    inv_structure_matrix::MM2
    "the fraction of variance to be retained after truncating singular values (1 implies no truncation)"
    retained_variance::FT
end



# ...struct VariationalAutoEncoder  <: DataContainerProcessor end

# PDCProcessors
struct DataInformedReducer <: PairedDataContainerProcessor end

struct LikelihoodInformedReducer <: PairedDataContainerProcessor end

struct CanonicalCorrelationReducer <: PairedDataContainerProcessor end


####


# generic function to build and encode data
function initialize_and_encode_data!(
    dcp::DCP,
    data::MM,
    obs_noise_cov::USorM,
) where {DCP <: DataContainerProcessor, USorM <: Union{UniformScaling, AbstractMatrix}, MM <: AbstractMatrix}
    initialize_processor!(dcp, data, obs_noise_cov)
    return encode_data(dcp, data, obs_noise_cov)
end

initialize_and_encode_data!(
    dcp::DCP,
    data::MM,
    obs_noise_cov::USorM,
    apply_to::AS,
) where {
    DCP <: DataContainerProcessor,
    MM <: AbstractMatrix,
    USorM <: Union{UniformScaling, AbstractMatrix},
    AS <: AbstractString,
} = initialize_and_encode_data!(dcp, data, obs_noise_cov)

decode_data(
    dcp::DCP,
    data::MM,
    obs_noise_cov::USorM,
    apply_to::AS,
) where {
    DCP <: DataContainerProcessor,
    MM <: AbstractMatrix,
    USorM <: Union{UniformScaling, AbstractMatrix},
    AS <: AbstractString,
} = decode_data(dcp, data, obs_noise_cov)

function initialize_and_encode_data!(
    dcp::PDCP,
    data,
    obs_noise_cov::USorM,
    apply_to::AS,
) where {PDCP <: PairedDataContainerProcessor, USorM <: Union{UniformScaling, AbstractMatrix}, AS <: AbstractString}
    input_data, output_data = data
    initialize_processor!(dcp, input_data, output_data, obs_noise_cov, apply_to)
    return encode_data(dcp, input_data, output_data, obs_noise_cov, apply_to)
end

function decode_data(
    dcp::PDCP,
    data,
    obs_noise_cov::USorM,
    apply_to::AS,
) where {PDCP <: PairedDataContainerProcessor, USorM <: Union{UniformScaling, AbstractMatrix}, AS <: AbstractString}
    input_data, output_data = data
    return decode_data(dcp, input_data, output_data, obs_noise_cov, apply_to)
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
        if isa(processor, DataContainerProcessor)
            if apply_to == "in"
                func = x -> get_inputs(x)
                push!(encoder_schedule, (processor, func, apply_to))

            elseif apply_to == "out"
                func = x -> get_outputs(x)
                push!(encoder_schedule, (processor, func, apply_to))

            elseif apply_to == "in_and_out"
                func1 = x -> get_inputs(x)
                func2 = x -> get_outputs(x)
                push!(encoder_schedule, (processor, func1, "in"))
                push!(encoder_schedule, (deepcopy(processor), func2, "out"))
            else
                @warn(
                    "Expected schedule keywords ∈ {\"in\",\"out\",\"in_and_out\"}. Received $(str), ignoring processor $(processor)..."
                )
            end

            # extract all the data (needed for the paired processor, but still pass through what you apply to)
        elseif isa(processor, PairedDataContainerProcessor)
            if apply_to ∈ ["in", "out"]
                func = x -> (get_inputs(x), get_outputs(x))
                push!(encoder_schedule, (processor, func, apply_to))
            elseif apply_to == "in_and_out"
                func = x -> (get_inputs(x), get_outputs(x))
                push!(encoder_schedule, (processor, func, "in"))
                push!(encoder_schedule, (deepcopy(processor), func, "out"))
            else
                @warn(
                    "Expected schedule keywords ∈ {\"in\",\"out\",\"in_and_out\"}. Received $(str), ignoring processor $(processor)..."
                )
            end
        end
    end

    return encoder_schedule
end

function encode_with_schedule(
    encoder_schedule::VV,
    io_pairs::PDC,
    prior_cov_in::USorM1,
    obs_noise_cov_in::USorM2,
) where {
    VV <: AbstractVector,
    PDC <: PairedDataContainer,
    USorM1 <: Union{UniformScaling, AbstractMatrix},
    USorM2 <: Union{UniformScaling, AbstractMatrix},
}

    processed_io_pairs = deepcopy(io_pairs)
    processed_prior_cov = deepcopy(prior_cov_in)
    processed_obs_noise_cov = deepcopy(obs_noise_cov_in)

    # apply_to is the string "in", "out" etc.
    for (processor, extract_data, apply_to) in encoder_schedule
        @info "Encoding data: $(apply_to) with $(processor)"
        if apply_to == "in"
            structure_matrix = processed_prior_cov
        elseif apply_to == "out"
            structure_matrix = processed_obs_noise_cov
        end
        processed = initialize_and_encode_data!(processor, extract_data(processed_io_pairs), structure_matrix, apply_to)
        if apply_to == "in"
            processed_prior_cov = encode_structure_matrix(processor, structure_matrix)
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            processed_obs_noise_cov = encode_structure_matrix(processor, structure_matrix)
            processed_io_pairs = PairedDataContainer(get_inputs(processed_io_pairs), processed)
        end
    end

    return processed_io_pairs, processed_prior_cov, processed_obs_noise_cov
end

function decode_with_schedule(
    encoder_schedule::VV,
    io_pairs::PDC,
    prior_cov_in::USorM1,
    obs_noise_cov_in::USorM2,
) where {
    VV <: AbstractVector,
    USorM1 <: Union{UniformScaling, AbstractMatrix},
    USorM2 <: Union{UniformScaling, AbstractMatrix},
    PDC <: PairedDataContainer,
}

    processed_io_pairs = deepcopy(io_pairs)
    processed_prior_cov = deepcopy(prior_cov_in)
    processed_obs_noise_cov = deepcopy(obs_noise_cov_in)

    # apply_to is the string "in", "out" etc.
    for idx in reverse(eachindex(encoder_schedule))
        (processor, extract_data, apply_to) = encoder_schedule[idx]
        if apply_to == "in"
            structure_matrix = processed_prior_cov
        elseif apply_to == "out"
            structure_matrix = processed_obs_noise_cov
        end

        @info "Decoding data: $(apply_to) with $(processor)"
        processed = decode_data(processor, extract_data(processed_io_pairs), structure_matrix, apply_to)

        if apply_to == "in"
            processed_prior_cov = decode_structure_matrix(processor, structure_matrix)
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            processed_obs_noise_cov = decode_structure_matrix(processor, structure_matrix)
            processed_io_pairs = PairedDataContainer(get_inputs(processed_io_pairs), processed)
        end
    end

    return processed_io_pairs, processed_prior_cov, processed_obs_noise_cov
end




end # module
