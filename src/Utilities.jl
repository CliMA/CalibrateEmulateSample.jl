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
export UnivariateAffineScaling, AffineScaler, QuartileScaling, MinMaxScaling, ZScoreScaling
export quartile_scale, minmax_scale, zscore_scale
export Decorrelater, decorrelate_sample_cov, decorrelate_structure_mat, decorrelate
export get_type,
    get_shift, get_scale, get_data_mean, get_encoder_mat, get_decoder_mat, get_retain_var, get_decorrelate_with
export create_encoder_schedule, encode_with_schedule, decode_with_schedule
export initialize_processor!,
    initialize_and_encode_data!, encode_data, decode_data, encode_structure_matrix, decode_structure_matrix

export CanonicalCorrelation
export canonical_correlation
export get_apply_to



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

The AffineScaler{T} will create an encoding of the data_container via affine transformations.

Different methods `T` will build different transformations:
- [`quartile_scale`](@ref) : creates `QuartileScaling`,
- [`minmax_scale`](@ref) : creates `MinMaxScaling`
- [`zscore_scale`](@ref) : creates `ZScoreScaling`

and are accessed with [`get_type`](@ref)
"""
struct AffineScaler{T, VV <: AbstractVector} <: DataContainerProcessor
    "storage for the shift applied to data"
    shift::VV
    "storage for the scaling"
    scale::VV
end

"""
$(TYPEDSIGNATURES)

Constructs `AffineScaler{QuartileScaling}` processor.
As part of an encoder schedule, it will apply the transform ``\\frac{x - Q2(x)}{Q3(x) - Q1(x)}`` to each data dimension.
Also known as "robust scaling"
"""
quartile_scale() = AffineScaler(QuartileScaling)

"""
$(TYPEDSIGNATURES)

Constructs `AffineScaler{MinMaxScaling}` processor.
As part of an encoder schedule, this will apply the transform ``\\frac{x - \\min(x)}{\\max(x) - \\min(x)}`` to each data dimension.
"""
minmax_scale() = AffineScaler(MinMaxScaling)

"""
$(TYPEDSIGNATURES)

Constructs `AffineScaler{ZScoreScaling}` processor.
As part of an encoder schedule, this will apply the transform ``\\frac{x-\\mu}{\\sigma}``, (where ``x\\sim N(\\mu,\\sigma)``), to each data dimension.
For multivariate standardization, see [`Decorrelater`](@ref) 
"""
zscore_scale() = AffineScaler(ZScoreScaling)

AffineScaler(::Type{UAS}) where {UAS <: UnivariateAffineScaling} =
    AffineScaler{UAS, Vector{Float64}}(Float64[], Float64[])

"""
$(TYPEDSIGNATURES)

Gets the UnivariateAffineScaling type `T`
"""
get_type(as::AffineScaler{T}) where {T} = T

"""
$(TYPEDSIGNATURES)

Gets the `shift` field of the `AffineScaler`
"""
get_shift(as::AffineScaler) = as.shift

"""
$(TYPEDSIGNATURES)

Gets the `scale` field of the `AffineScaler`
"""
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
    minmax_mat = reduce(hcat, minmax_vec) # 2 rows: min max
    append!(get_shift(as), minmax_mat[1, :])
    append!(get_scale(as), (minmax_mat[2, :] - minmax_mat[1, :]))
end

function initialize_processor!(
    as::AffineScaler,
    data::MM,
    T::Type{ZSS},
) where {MM <: AbstractMatrix, ZSS <: ZScoreScaling}
    stat_vec = [[mean(dd), std(dd)] for dd in eachrow(data)]
    stat_mat = reduce(hcat, stat_vec) # 2 rows: mean, std
    append!(get_shift(as), stat_mat[1, :])
    append!(get_scale(as), stat_mat[2, :])
end

function initialize_processor!(as::AffineScaler, data::MM) where {MM <: AbstractMatrix}
    if length(get_shift(as)) == 0
        T = get_type(as)
        initialize_processor!(as, data, T)
    end
end

"""
$(TYPEDSIGNATURES)

Apply the `AffineScaler` encoder, on a columns-are-data matrix
"""
function encode_data(as::AffineScaler, data::MM) where {MM <: AbstractMatrix}
    out = deepcopy(data)
    for i in 1:size(out, 1)
        out[i, :] .-= get_shift(as)[i]
        out[i, :] /= get_scale(as)[i]
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Apply the `AffineScaler` decoder, on a columns-are-data matrix
"""
function decode_data(as::AffineScaler, data::MM) where {MM <: AbstractMatrix}
    out = deepcopy(data)
    for i in 1:size(out, 1)
        out[i, :] *= get_scale(as)[i]
        out[i, :] .+= get_shift(as)[i]
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Computes and populates the `shift` and `scale` fields for the `AffineScaler`
"""
initialize_processor!(as::AffineScaler, data::MM, structure_matrix) where {MM <: AbstractMatrix} =
    initialize_processor!(as, data)


"""
$(TYPEDSIGNATURES)

Apply the `AffineScaler` encoder to a provided structure matrix
"""
function encode_structure_matrix(as::AffineScaler, structure_matrix::MM) where {MM <: AbstractMatrix}
    return Diagonal(1 ./ get_scale(as)) * structure_matrix * Diagonal(1 ./ get_scale(as))
end

"""
$(TYPEDSIGNATURES)

Apply the `AffineScaler` decoder to a provided structure matrix
"""
function decode_structure_matrix(as::AffineScaler, enc_structure_matrix::MM) where {MM <: AbstractMatrix}
    return Diagonal(get_scale(as)) * enc_structure_matrix * Diagonal(get_scale(as))
end


"""
$(TYPEDEF)

Decorrelate the data via taking an SVD decomposition and projecting onto the singular-vectors. 

Preferred construction is with the methods
- [`decorrelate_structure_mat`](@ref)
- [`decorrelate_sample_cov`](@ref)
- [`decorrelate`](@ref) 

For `decorrelate_structure_mat`:
The SVD is taken over a structure matrix (e.g., `prior_cov` for inputs, `obs_noise_cov` for outputs). The structure matrix will become exactly `I` after processing.

For `decorrelate_sample_cov`:
The SVD is taken over the estimated covariance of the data. The data samples will have a `Normal(0,I)` distribution after processing,

For `decorrelate(;decorrelate_with="combined")` (default):
The SVD is taken to be the sum of structure matrix and estimated covariance. This may be more robust to ill-specification of structure matrix, or poor estimation of the sample covariance.

# Fields
$(TYPEDFIELDS)
"""
struct Decorrelater{VV1, VV2, VV3, FT, AS <: AbstractString} <: DataContainerProcessor
    "storage for the data mean"
    data_mean::VV1
    "the matrix used to perform encoding"
    encoder_mat::VV2
    "the inverse of the the matrix used to perform encoding"
    decoder_mat::VV3
    "the fraction of variance to be retained after truncating singular values (1 implies no truncation)"
    retain_var::FT
    "Switch to choose what form of matrix to use to decorrelate the data"
    decorrelate_with::AS
end

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelater` struct. Users can add optional keyword arguments:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
- `decorrelate_with` [=`"structure_matrix"`]: from which matrix do we provide subspace directions, options are
  - `"structure_mat"`, see [`decorrelate_structure_mat`](@ref)
  - `"sample_cov"`, see [`decorrelate_sample_cov`](@ref)
  - `"combined"`, sums the `"sample_cov"` and `"structure_mat"` matrices
"""
decorrelate(; retain_var::FT = Float64(1.0), decorrelate_with = "combined") where {FT} =
    Decorrelater([], [], [], min(max(retain_var, FT(0)), FT(1)), decorrelate_with)

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelater` struct, setting decorrelate_with = "sample_cov". Encoding data with this will ensure that the distribution of data samples after encoding will be `Normal(0,I)`. One can additionally add keywords:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
"""
decorrelate_sample_cov(; retain_var::FT = Float64(1.0)) where {FT} =
    Decorrelater([], [], [], min(max(retain_var, FT(0)), FT(1)), "sample_cov")

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelater` struct, setting decorrelate_with = "structure_mat". This encoding will transform a provided structure matrix into `I`. One can additionally add keywords:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
"""
decorrelate_structure_mat(; retain_var::FT = Float64(1.0)) where {FT} =
    Decorrelater([], [], [], min(max(retain_var, FT(0)), FT(1)), "structure_mat")

"""
$(TYPEDSIGNATURES)

returns the `data_mean` field of the `Decorrelater`.
"""
get_data_mean(dd::Decorrelater) = dd.data_mean

"""
$(TYPEDSIGNATURES)

returns the `encoder_mat` field of the `Decorrelater`.
"""
get_encoder_mat(dd::Decorrelater) = dd.encoder_mat

"""
$(TYPEDSIGNATURES)

returns the `decoder_mat` field of the `Decorrelater`.
"""
get_decoder_mat(dd::Decorrelater) = dd.decoder_mat

"""
$(TYPEDSIGNATURES)

returns the `retain_var` field of the `Decorrelater`.
"""
get_retain_var(dd::Decorrelater) = dd.retain_var

"""
$(TYPEDSIGNATURES)

returns the `decorrelate_with` field of the `Decorrelater`.
"""
get_decorrelate_with(dd::Decorrelater) = dd.decorrelate_with

function Base.show(io::IO, dd::Decorrelater)
    out = "Decorrelater"
    if get_retain_var(dd) < 1.0
        out *= ": retain_var=$(get_retain_var(dd)) "
    end
    out *= ": decorrelate_with=$(get_decorrelate_with(dd)) "
    print(io, out)
end

"""
$(TYPEDSIGNATURES)

Computes and populates the `data_mean` and `encoder_mat` and `decoder_mat` fields for the `Decorrelater`
"""
function initialize_processor!(
    dd::Decorrelater,
    data::MM,
    structure_matrix::USorM,
) where {MM <: AbstractMatrix, USorM <: Union{UniformScaling, AbstractMatrix}}
    if length(get_data_mean(dd)) == 0
        append!(get_data_mean(dd), mean(data, dims = 2))
    end

    if length(get_encoder_mat(dd)) == 0

        # Can do tsvd here for large matrices
        decorrelate_with = get_decorrelate_with(dd)
        if decorrelate_with == "structure_mat"
            svdA = svd(structure_matrix)
            rk = rank(structure_matrix)
        elseif decorrelate_with == "sample_cov"
            cd = cov(data, dims = 2)
            svdA = svd(cd)
            rk = rank(cd)
        elseif decorrelate_with == "combined"
            spluscd = structure_matrix + cov(data, dims = 2)
            svdA = svd(spluscd)
            rk = rank(spluscd)
        else
            throw(
                ArgumentError(
                    "Keyword `decorrelate_with` must be taken from [\"sample_cov\", \"structure_mat\", \"combined\"]. Received $(decorrelate_with)",
                ),
            )
        end
        ret_var = get_retain_var(dd)
        if ret_var < 1.0
            sv_cumsum = cumsum(svdA.S .^ 2) / sum(svdA.S .^ 2) # variance contributions are (sing_val)^2
            trunc = minimum(findall(x -> (x > ret_var), sv_cumsum))
            @info "    truncating at $(trunc)/$(length(sv_cumsum)) retaining $(100.0*sv_cumsum[trunc])% of the variance of the structure matrix"
        else
            trunc = rk
        end
        sqrt_inv_sv = Diagonal(1.0 ./ sqrt.(svdA.S[1:trunc]))
        sqrt_sv = Diagonal(sqrt.(svdA.S[1:trunc]))

        # as we have svd of cov-matrix we can use U or Vt
        encoder_mat = sqrt_inv_sv * svdA.Vt[1:trunc, :]
        decoder_mat = svdA.Vt[1:trunc, :]' * sqrt_sv

        push!(get_encoder_mat(dd), encoder_mat)
        push!(get_decoder_mat(dd), decoder_mat)
    end
end


"""
$(TYPEDSIGNATURES)

Apply the `Decorrelater` encoder, on a columns-are-data matrix
"""
function encode_data(dd::Decorrelater, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(dd)
    encoder_mat = get_encoder_mat(dd)[1]
    return encoder_mat * (data .- data_mean)
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelater` decoder, on a columns-are-data matrix
"""
function decode_data(dd::Decorrelater, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(dd)
    decoder_mat = get_decoder_mat(dd)[1]
    return decoder_mat * data .+ data_mean
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelater` encoder to a provided structure matrix
"""
function encode_structure_matrix(dd::Decorrelater, structure_matrix::MM) where {MM <: AbstractMatrix}
    encoder_mat = get_encoder_mat(dd)[1]
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelater` decoder to a provided structure matrix
"""
function decode_structure_matrix(dd::Decorrelater, enc_structure_matrix::MM) where {MM <: AbstractMatrix}
    decoder_mat = get_decoder_mat(dd)[1]
    return decoder_mat * enc_structure_matrix * decoder_mat'
end

# ...struct VariationalAutoEncoder  <: DataContainerProcessor end

# PDCProcessors
# struct InverseProblemInformed <: PairedDataContainerProcessor end
# struct LikelihoodInformed <: PairedDataContainerProcessor end


"""
$(TYPEDEF)

Uses both input and output data to learn a subspace of maximal correlation between inputs and outputs. The subspace for a pair (X,Y) will be of size minimum(rank(X),rank(Y)), computed using SVD-based method
e.g. See e.g., https://numerical.recipes/whp/notes/CanonCorrBySVD.pdf

Preferred construction is with the [`canonical_correlation`](@ref) method

# Fields
$(TYPEDFIELDS)
"""
struct CanonicalCorrelation{VV1, VV2, VV3, FT, VV4} <: PairedDataContainerProcessor
    "storage for the input or output data mean"
    data_mean::VV1
    "the encoding matrix of input or output canonical correlations"
    encoder_mat::VV2
    "the decoding matrix of input or output canonical correlations"
    decoder_mat::VV3
    "the fraction of variance to be retained after truncating singular values (1 implies no truncation)"
    retain_var::FT
    "Stores whether this is an input or output encoder (vector with string \"in\" or \"out\")"
    apply_to::VV4
end

"""
$(TYPEDSIGNATURES)

Constructs the `CanonicalCorrelation` struct. Can optionally provide the keyword
- `retain_var`[=1.0]: to project onto the leading singular vectors (of the input-output product) such that `retain_var` variance is retained. 
"""
canonical_correlation(; retain_var = Float64(1.0)) =
    CanonicalCorrelation(Any[], Any[], Any[], retain_var, AbstractString[])

"""
$(TYPEDSIGNATURES)

returns the `data_mean` field of the `CanonicalCorrelation`.
"""
get_data_mean(cc::CanonicalCorrelation) = cc.data_mean

"""
$(TYPEDSIGNATURES)

returns the `encoder_mat` field of the `CanonicalCorrelation`.
"""
get_encoder_mat(cc::CanonicalCorrelation) = cc.encoder_mat

"""
$(TYPEDSIGNATURES)

returns the `decoder_mat` field of the `CanonicalCorrelation`.
"""
get_decoder_mat(cc::CanonicalCorrelation) = cc.decoder_mat

"""
$(TYPEDSIGNATURES)

returns the `retain_var` field of the `CanonicalCorrelation`.
"""
get_retain_var(cc::CanonicalCorrelation) = cc.retain_var

"""
$(TYPEDSIGNATURES)

returns the `apply_to` field of the `CanonicalCorrelation`.
"""
get_apply_to(cc::CanonicalCorrelation) = cc.apply_to

function Base.show(io::IO, cc::CanonicalCorrelation)
    if get_retain_var(cc) < 1.0
        out = "CanonicalCorrelation: retain_var=$(get_retain_var(cc))"
    else
        out = "CanonicalCorrelation"
    end
    print(io, out)
end


function initialize_processor!(
    cc::CanonicalCorrelation,
    in_data::MM,
    out_data::MM,
    apply_to::AS,
) where {MM <: AbstractMatrix, AS <: AbstractString}

    if length(get_apply_to(cc)) == 0
        push!(get_apply_to(cc), apply_to)
    end

    if length(get_data_mean(cc)) == 0
        if apply_to == "in"
            append!(get_data_mean(cc), mean(in_data, dims = 2))
        elseif apply_to == "out"
            append!(get_data_mean(cc), mean(out_data, dims = 2))
        end
    end

    if length(get_encoder_mat(cc)) == 0

        if size(in_data, 2) < size(in_data, 1) || size(out_data, 2) < size(out_data, 1)
            throw(
                ArgumentError(
                    "CanonicalCorrelation implementation not defined for # data samples < dimensions, please obtain more samples, or perform prior dimension reduction approaches until this is satisfied",
                ),
            )
        end

        # Individually decompose in and out
        # Want to use the nonsquare singular vector
        svdi = svd(in_data .- mean(in_data, dims = 2))
        svdo = svd(out_data .- mean(out_data, dims = 2))
        # determine regime

        # ensure we non-square sv (in_mat = (in_dim x n_samples), out_mat = (out_dim x n_samples))
        in_mat_sq, in_mat_nonsq = (size(svdi.U, 1) == size(svdi.U, 2)) ? (svdi.U, svdi.Vt) : (svdi.Vt, svdi.U)
        out_mat_sq, out_mat_nonsq = (size(svdo.U, 1) == size(svdo.U, 2)) ? (svdo.U, svdo.Vt) : (svdo.Vt, svdo.U)

        svdio = svd(out_mat_nonsq * in_mat_nonsq')

        # retain variance
        ret_var = get_retain_var(cc)
        if ret_var < 1.0
            sv_cumsum = cumsum(svdio.S .^ 2) / sum(svdio.S .^ 2) # variance contributions are (sing_val)^2
            trunc = minimum(findall(x -> (x > ret_var), sv_cumsum))
            @info "    truncating at $(trunc)/$(length(sv_cumsum)) retaining $(100.0*sv_cumsum[trunc])% of the variance in the joint space"
        else
            trunc = min(rank(in_data), rank(out_data))
        end

        if apply_to == "in"
            in_dim = size(in_data, 1)
            svdio_mat = (size(svdio.U, 1) == in_dim) ? svdio.U : svdio.V
            # mat' * Sx⁻¹ * Uxt
            encoder_mat = svdio_mat[:, 1:trunc]' * Diagonal(1 ./ svdi.S) * in_mat_sq'
            decoder_mat = in_mat_sq * Diagonal(svdi.S) * svdio_mat[:, 1:trunc]
        else
            apply_to == "out"
            out_dim = size(out_data, 1)
            svdio_mat = (size(svdio.U, 1) == out_dim) ? svdio.U : svdio.V
            # Vt * Sy⁻¹ * Uyt
            encoder_mat = svdio_mat[:, 1:trunc]' * Diagonal(1 ./ svdo.S) * out_mat_sq'
            decoder_mat = out_mat_sq * Diagonal(svdo.S) * svdio_mat[:, 1:trunc]
        end

        push!(get_encoder_mat(cc), encoder_mat)
        push!(get_decoder_mat(cc), decoder_mat)

        # Note: To check CCA: 
        # u = in_encoder * (in - mean(in))
        # v = out_encoder * (out - mean(out))
        # u * u' = v * v' = I, 
        # v * u' = u * v' = Diagonal(svdio.S[1:trunc])
    end
end

"""
$(TYPEDSIGNATURES)

Computes and populates the `data_mean`, `encoder_mat`, `decoder_mat` and `apply_to` fields for the `CanonicalCorrelation`
"""
initialize_processor!(
    cc::CanonicalCorrelation,
    in_data::MM,
    out_data::MM,
    structure_matrix,
    apply_to::AS,
) where {MM <: AbstractMatrix, AS <: AbstractString} = initialize_processor!(cc, in_data, out_data, apply_to)


"""
$(TYPEDSIGNATURES)

Apply the `CanonicalCorrelation` encoder, on a columns-are-data matrix
"""
function encode_data(cc::CanonicalCorrelation, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(cc)
    encoder_mat = get_encoder_mat(cc)[1]
    return encoder_mat * (data .- data_mean)
end

"""
$(TYPEDSIGNATURES)

Apply the `CanonicalCorrelation` decoder, on a columns-are-data matrix
"""
function decode_data(cc::CanonicalCorrelation, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(cc)
    decoder_mat = get_decoder_mat(cc)[1]
    return decoder_mat * data .+ data_mean
end

"""
$(TYPEDSIGNATURES)

Apply the `CanonicalCorrelation` encoder to a provided structure matrix
"""
function encode_structure_matrix(cc::CanonicalCorrelation, structure_matrix::MM) where {MM <: AbstractMatrix}
    encoder_mat = get_encoder_mat(cc)[1]
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `CanonicalCorrelation` decoder to a provided structure matrix
"""
function decode_structure_matrix(cc::CanonicalCorrelation, enc_structure_matrix::MM) where {MM <: AbstractMatrix}
    decoder_mat = get_decoder_mat(cc)[1]
    return decoder_mat * enc_structure_matrix * decoder_mat'
end





####


# generic functions to initialize, encode, and decode data. With Data processors or PairedData processors
function initialize_and_encode_data!(
    dcp::DCP,
    data::MM,
    structure_mat::USorM,
) where {DCP <: DataContainerProcessor, USorM <: Union{UniformScaling, AbstractMatrix}, MM <: AbstractMatrix}
    initialize_processor!(dcp, data, structure_mat)
    return encode_data(dcp, data)
end

"""
$(TYPEDSIGNATURES)

Initializes the `DataContainerProcessor` encoder (often requires data, and structure matrices), then encodes the provided columns-are-data matrix
"""
initialize_and_encode_data!(
    dcp::DCP,
    data::MM,
    structure_mat::USorM,
    apply_to::AS,
) where {
    DCP <: DataContainerProcessor,
    MM <: AbstractMatrix,
    USorM <: Union{UniformScaling, AbstractMatrix},
    AS <: AbstractString,
} = initialize_and_encode_data!(dcp, data, structure_mat)


"""
$(TYPEDSIGNATURES)

decodes the columns-are-data matrix with the processor.
"""
decode_data(
    dcp::DCP,
    data::MM,
    apply_to::AS,
) where {DCP <: DataContainerProcessor, MM <: AbstractMatrix, AS <: AbstractString} = decode_data(dcp, data)

"""
$(TYPEDSIGNATURES)

Initializes the `PairedDataContainerProcesser` encoder (often requires input & output data, and structure matrices), then encodes either the input or output data (pair of columns-are-data matrices)  based on `apply_to`.
"""
function initialize_and_encode_data!(
    dcp::PDCP,
    data,
    structure_mat::USorM,
    apply_to::AS,
) where {PDCP <: PairedDataContainerProcessor, USorM <: Union{UniformScaling, AbstractMatrix}, AS <: AbstractString}
    input_data, output_data = data
    initialize_processor!(dcp, input_data, output_data, structure_mat, apply_to)
    if apply_to == "in"
        return encode_data(dcp, input_data)
    elseif apply_to == "out"
        return encode_data(dcp, output_data)
    end
end

"""
$(TYPEDSIGNATURES)

decodes the input or output dat (pair of columns-are-data matrices) a with the processor, based on `apply_to`.
"""
function decode_data(dcp::PDCP, data, apply_to::AS) where {PDCP <: PairedDataContainerProcessor, AS <: AbstractString}
    input_data, output_data = data
    if apply_to == "in"
        return decode_data(dcp, input_data)
    elseif apply_to == "out"
        return decode_data(dcp, output_data)
    end
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
                    "Expected schedule keywords ∈ {\"in\",\"out\",\"in_and_out\"}. Received $(apply_to), ignoring processor $(processor)..."
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
                    "Expected schedule keywords ∈ {\"in\",\"out\",\"in_and_out\"}. Received $(apply_to), ignoring processor $(processor)..."
                )
            end
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
function encode_with_schedule(
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
    for (processor, extract_data, apply_to) in encoder_schedule
        @info "Initialize encoding of data: \"$(apply_to)\" with $(processor)"
        if apply_to == "in"
            structure_matrix = processed_input_structure_mat
        elseif apply_to == "out"
            structure_matrix = processed_output_structure_mat
        end
        processed = initialize_and_encode_data!(processor, extract_data(processed_io_pairs), structure_matrix, apply_to)
        if apply_to == "in"
            processed_input_structure_mat = encode_structure_matrix(processor, structure_matrix)
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            processed_output_structure_mat = encode_structure_matrix(processor, structure_matrix)
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

    if !(in_or_out ∈ ["in", "out"])
        throw(
            ArgumentError(
                "`in_or_out` must be either \"in\" (data is an input) or \"out\" (data is an output). Received $(in_or_out)",
            ),
        )
    end
    processed_container = deepcopy(data_container)

    # apply_to is the string "in", "out" etc.
    for (processor, extract_data, apply_to) in encoder_schedule
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
        throw(
            ArgumentError(
                "`in_or_out` must be either \"in\" (data is an input) or \"out\" (data is an output). Received $(in_or_out)",
            ),
        )
    end
    processed_structure_matrix = deepcopy(structure_matrix)

    # apply_to is the string "in", "out" etc.
    for (processor, extract_data, apply_to) in encoder_schedule
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
        (processor, extract_data, apply_to) = encoder_schedule[idx]
        processed = decode_data(processor, extract_data(processed_io_pairs), apply_to)

        if apply_to == "in"
            processed_input_structure_mat = decode_structure_matrix(processor, processed_input_structure_mat)
            processed_io_pairs = PairedDataContainer(processed, get_outputs(processed_io_pairs))
        elseif apply_to == "out"
            processed_output_structure_mat = decode_structure_matrix(processor, processed_output_structure_mat)
            processed_io_pairs = PairedDataContainer(get_inputs(processed_io_pairs), processed)
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

    if !(in_or_out ∈ ["in", "out"])
        throw(
            ArgumentError(
                "`in_or_out` must be either \"in\" (data is an input) or \"out\" (data is an output). Received $(in_or_out)",
            ),
        )
    end
    processed_container = deepcopy(data_container)

    # apply_to is the string "in", "out" etc.
    for idx in reverse(eachindex(encoder_schedule))
        (processor, extract_data, apply_to) = encoder_schedule[idx]
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

    if !(in_or_out ∈ ["in", "out"])
        throw(
            ArgumentError(
                "`in_or_out` must be either \"in\" (data is an input) or \"out\" (data is an output). Received $(in_or_out)",
            ),
        )
    end
    processed_structure_matrix = deepcopy(structure_matrix)

    # apply_to is the string "in", "out" etc.
    for idx in reverse(eachindex(encoder_schedule))
        (processor, extract_data, apply_to) = encoder_schedule[idx]
        if apply_to == in_or_out
            processed_structure_matrix = decode_structure_matrix(processor, processed_structure_matrix)
        end
    end

    return processed_structure_matrix
end



end # module
