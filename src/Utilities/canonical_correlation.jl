# included in Utilities.jl

export CanonicalCorrelation, canonical_correlation, get_apply_to
export get_shift, get_scale, get_data_mean, get_encoder_mat, get_decoder_mat, get_retain_var, get_decorrelate_with

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
canonical_correlation(; retain_var::FT = Float64(1.0)) where {FT} =
    CanonicalCorrelation(Any[], Any[], Any[], clamp(retain_var, FT(0), FT(1)), AbstractString[])

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

    out = "CanonicalCorrelation:"
    if length(get_apply_to(cc)) > 0
        out *= " apply_to=$(get_apply_to(cc)[1])"
    end
    if get_retain_var(cc) < 1.0
        out *= " retain_var=$(get_retain_var(cc))"
    end
    print(io, out)
end


function initialize_processor!(
    cc::CanonicalCorrelation,
    in_data::MM,
    out_data::MM,
    apply_to::AS,
) where {MM <: AbstractMatrix, AS <: AbstractString}

    if apply_to ∉ ["in", "out"]
        bad_apply_to(apply_to)
    end

    if length(get_apply_to(cc)) == 0
        push!(get_apply_to(cc), apply_to)
    end

    if length(get_data_mean(cc)) == 0
        if apply_to == "in"
            push!(get_data_mean(cc), vec(mean(in_data, dims = 2)))
        elseif apply_to == "out"
            push!(get_data_mean(cc), vec(mean(out_data, dims = 2)))
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
        svdi = svd(in_data .- mean(in_data, dims = 2))
        svdo = svd(out_data .- mean(out_data, dims = 2))

        # ensure correct shaping (in_mat = (in_dim x n_samples), out_mat = (out_dim x n_samples))
        in_mat_sq, in_mat_nonsq = (size(svdi.U, 1) == size(svdi.U, 2)) ? (svdi.U, svdi.Vt) : (svdi.Vt, svdi.U)
        out_mat_sq, out_mat_nonsq = (size(svdo.U, 1) == size(svdo.U, 2)) ? (svdo.U, svdo.Vt) : (svdo.Vt, svdo.U)

        svdio = svd(in_mat_nonsq * out_mat_nonsq')

        # retain variance
        ret_var = get_retain_var(cc)
        if ret_var < 1.0
            sv_cumsum = cumsum(svdio.S .^ 2) / sum(svdio.S .^ 2) # variance contributions are (sing_val)^2for these matrices
            trunc_val = minimum(findall(x -> (x > ret_var), sv_cumsum))
            @info "    truncating at $(trunc_val)/$(length(sv_cumsum)) retaining $(100.0*sv_cumsum[trunc_val])% of the variance in the joint space"
        else
            trunc_val = min(rank(in_data), rank(out_data))
        end
        in_dim = size(in_data, 1)
        in_svdio_mat, out_svdio_mat = (size(svdio.U, 1) == in_dim) ? (svdio.U, svdio.V) : (svdio.V, svdio.U')
        if apply_to == "in"
            # mat' * Sx⁻¹ * Uxt
            encoder_mat = in_svdio_mat[:, 1:trunc_val]' * Diagonal(1 ./ svdi.S) * in_mat_sq'
            decoder_mat = in_mat_sq * Diagonal(svdi.S) * in_svdio_mat[:, 1:trunc_val]

        elseif apply_to == "out"
            out_dim = size(out_data, 1)
            # Vt * Sy⁻¹ * Uyt
            encoder_mat = out_svdio_mat[:, 1:trunc_val]' * Diagonal(1 ./ svdo.S) * out_mat_sq'
            decoder_mat = out_mat_sq * Diagonal(svdo.S) * out_svdio_mat[:, 1:trunc_val]
        end

        push!(get_encoder_mat(cc), encoder_mat)
        push!(get_decoder_mat(cc), decoder_mat)

        # Note: To check CCA: 
        # u = in_encoder * (in_data .- mean(in_data, dims=2))
        # v = out_encoder * (out_data .- mean(out_data, dims=2))
        # u * u' = v * v' = I, 
        # v * u' = u * v' = Diagonal(svdio.S[1:trunc_val])
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
    input_structure_matrix,
    output_structure_matrix,
    apply_to::AS,
) where {MM <: AbstractMatrix, AS <: AbstractString} = initialize_processor!(cc, in_data, out_data, apply_to)


"""
$(TYPEDSIGNATURES)

Apply the `CanonicalCorrelation` encoder, on a columns-are-data matrix
"""
function encode_data(cc::CanonicalCorrelation, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(cc)[1]
    encoder_mat = get_encoder_mat(cc)[1]
    return encoder_mat * (data .- data_mean)
end

"""
$(TYPEDSIGNATURES)

Apply the `CanonicalCorrelation` decoder, on a columns-are-data matrix
"""
function decode_data(cc::CanonicalCorrelation, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(cc)[1]
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
