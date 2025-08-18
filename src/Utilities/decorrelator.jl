# included in Utilities.jl

export Decorrelator, decorrelate_sample_cov, decorrelate_structure_mat, decorrelate
export get_data_mean, get_encoder_mat, get_decoder_mat, get_retain_var, get_decorrelate_with

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
struct Decorrelator{VV1, VV2, VV3, FT, AS <: AbstractString} <: DataContainerProcessor
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
    ""
    structure_mat_name::Union{Nothing, Symbol}
end

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelator` struct. Users can add optional keyword arguments:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
- `decorrelate_with` [=`"combined"`]: from which matrix do we provide subspace directions, options are
  - `"structure_mat"`, see [`decorrelate_structure_mat`](@ref)
  - `"sample_cov"`, see [`decorrelate_sample_cov`](@ref)
  - `"combined"`, sums the `"sample_cov"` and `"structure_mat"` matrices
"""
decorrelate(; retain_var::FT = Float64(1.0), decorrelate_with = "combined", structure_mat_name = nothing) where {FT} =
    Decorrelator([], [], [], clamp(retain_var, FT(0), FT(1)), decorrelate_with, structure_mat_name)

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelator` struct, setting decorrelate_with = "sample_cov". Encoding data with this will ensure that the distribution of data samples after encoding will be `Normal(0,I)`. One can additionally add keywords:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
"""
decorrelate_sample_cov(; retain_var::FT = Float64(1.0)) where {FT} =
    Decorrelator([], [], [], clamp(retain_var, FT(0), FT(1)), "sample_cov", nothing)

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelator` struct, setting decorrelate_with = "structure_mat". This encoding will transform a provided structure matrix into `I`. One can additionally add keywords:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
"""
decorrelate_structure_mat(; retain_var::FT = Float64(1.0), structure_mat_name = nothing) where {FT} =
    Decorrelator([], [], [], clamp(retain_var, FT(0), FT(1)), "structure_mat", structure_mat_name)

"""
$(TYPEDSIGNATURES)

returns the `data_mean` field of the `Decorrelator`.
"""
get_data_mean(dd::Decorrelator) = dd.data_mean

"""
$(TYPEDSIGNATURES)

returns the `encoder_mat` field of the `Decorrelator`.
"""
get_encoder_mat(dd::Decorrelator) = dd.encoder_mat

"""
$(TYPEDSIGNATURES)

returns the `decoder_mat` field of the `Decorrelator`.
"""
get_decoder_mat(dd::Decorrelator) = dd.decoder_mat

"""
$(TYPEDSIGNATURES)

returns the `retain_var` field of the `Decorrelator`.
"""
get_retain_var(dd::Decorrelator) = dd.retain_var

"""
$(TYPEDSIGNATURES)

returns the `decorrelate_with` field of the `Decorrelator`.
"""
get_decorrelate_with(dd::Decorrelator) = dd.decorrelate_with

function Base.show(io::IO, dd::Decorrelator)
    out = "Decorrelator"
    out *= ": decorrelate_with=$(get_decorrelate_with(dd))"
    if get_retain_var(dd) < 1.0
        out *= ", retain_var=$(get_retain_var(dd))"
    end
    print(io, out)
end

"""
$(TYPEDSIGNATURES)

Computes and populates the `data_mean` and `encoder_mat` and `decoder_mat` fields for the `Decorrelator`
"""
function initialize_processor!(
    dd::Decorrelator,
    data::MM,
    structure_matrices::Dict{Symbol, <:StructureMatrix},
    ::Dict{Symbol, <:StructureVector},
) where {MM <: AbstractMatrix}
    if length(get_data_mean(dd)) == 0
        push!(get_data_mean(dd), vec(mean(data, dims = 2)))
    end

    if length(get_encoder_mat(dd)) == 0

        # Can do tsvd here for large matrices
        decorrelate_with = get_decorrelate_with(dd)
        if decorrelate_with == "structure_mat"
            structure_matrix = get_structure_mat(structure_matrices, dd.structure_mat_name)
            if isa(structure_matrix, UniformScaling)
                data_dim = size(data, 1)
                svdA = svd(structure_matrix(data_dim))
                rk = data_dim
            else
                svdA = svd(structure_matrix)
                rk = rank(structure_matrix)
            end
        elseif decorrelate_with == "sample_cov"
            cd = cov(data, dims = 2)
            svdA = svd(cd)
            rk = rank(cd)
        elseif decorrelate_with == "combined"
            structure_matrix = get_structure_mat(structure_matrices, dd.structure_mat_name)
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
            sv_cumsum = cumsum(svdA.S) / sum(svdA.S) # variance contributions are (sing_val) for these matrices
            trunc_val = minimum(findall(x -> (x > ret_var), sv_cumsum))
            @info "    truncating at $(trunc_val)/$(length(sv_cumsum)) retaining $(100.0*sv_cumsum[trunc_val])% of the variance of the structure matrix"
        else
            trunc_val = rk
            if rk < size(data, 1)
                @info "    truncating at $(trunc_val)/$(size(data,1)), as low-rank data detected"
            end
        end

        sqrt_inv_sv = Diagonal(1.0 ./ sqrt.(svdA.S[1:trunc_val]))
        sqrt_sv = Diagonal(sqrt.(svdA.S[1:trunc_val]))
        # as we have svd of cov-matrix we can use U or Vt
        encoder_mat = sqrt_inv_sv * svdA.Vt[1:trunc_val, :]
        decoder_mat = svdA.Vt[1:trunc_val, :]' * sqrt_sv

        push!(get_encoder_mat(dd), encoder_mat)
        push!(get_decoder_mat(dd), decoder_mat)
    end
end


"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` encoder, on a columns-are-data matrix
"""
function encode_data(dd::Decorrelator, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(dd)[1]
    encoder_mat = get_encoder_mat(dd)[1]
    return encoder_mat * (data .- data_mean)
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` decoder, on a columns-are-data matrix
"""
function decode_data(dd::Decorrelator, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(dd)[1]
    decoder_mat = get_decoder_mat(dd)[1]
    return decoder_mat * data .+ data_mean
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` encoder to a provided structure matrix
"""
function encode_structure_matrix(dd::Decorrelator, structure_matrix::SM) where {SM <: StructureMatrix}
    encoder_mat = get_encoder_mat(dd)[1]
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` decoder to a provided structure matrix
"""
function decode_structure_matrix(dd::Decorrelator, enc_structure_matrix::SM) where {SM <: StructureMatrix}
    decoder_mat = get_decoder_mat(dd)[1]
    return decoder_mat * enc_structure_matrix * decoder_mat'
end
