# included in Utilities.jl

export UnivariateAffineScaling, ElementwiseScaler, QuartileScaling, MinMaxScaling, ZScoreScaling
export quartile_scale, minmax_scale, zscore_scale
export get_type,
    get_shift, get_scale, get_data_encoder_mat, get_data_decoder_mat, get_struct_encoder_mat, get_struct_decoder_mat

"""
$(TYPEDEF)

The ElementwiseScaler{T} will create an encoding of the data_container via elementwise affine transformations.

Different methods `T` will build different transformations:
- [`quartile_scale`](@ref) : creates `QuartileScaling`,
- [`minmax_scale`](@ref) : creates `MinMaxScaling`
- [`zscore_scale`](@ref) : creates `ZScoreScaling`

and are accessed with [`get_type`](@ref)
"""
struct ElementwiseScaler{
    T,
    VV <: AbstractVector,
    VV2 <: AbstractVector,
    VV3 <: AbstractVector,
    VV4 <: AbstractVector,
    VV5 <: AbstractVector,
} <: DataContainerProcessor
    "storage for the shift applied to data"
    shift::VV
    "storage for the scaling"
    scale::VV
    "the matrix used to perform encoding of data samples"
    data_encoder_mat::VV2
    "the inverse of the the matrix used to perform encoding of data samples"
    data_decoder_mat::VV3
    "the matrix used to perform encoding of structure_matrices"
    struct_encoder_mat::VV4
    "the inverse of the the matrix used to perform encoding of structure_matrices"
    struct_decoder_mat::VV5
end

abstract type UnivariateAffineScaling end
abstract type QuartileScaling <: UnivariateAffineScaling end
abstract type MinMaxScaling <: UnivariateAffineScaling end
abstract type ZScoreScaling <: UnivariateAffineScaling end

"""
$(TYPEDSIGNATURES)

Constructs `ElementwiseScaler{QuartileScaling}` processor.
As part of an encoder schedule, it will apply the transform ``\\frac{x - Q2(x)}{Q3(x) - Q1(x)}`` to each data dimension.
Also known as "robust scaling"
"""
quartile_scale() = ElementwiseScaler(QuartileScaling)

"""
$(TYPEDSIGNATURES)

Constructs `ElementwiseScaler{MinMaxScaling}` processor.
As part of an encoder schedule, this will apply the transform ``\\frac{x - \\min(x)}{\\max(x) - \\min(x)}`` to each data dimension.
"""
minmax_scale() = ElementwiseScaler(MinMaxScaling)

"""
$(TYPEDSIGNATURES)

Constructs `ElementwiseScaler{ZScoreScaling}` processor.
As part of an encoder schedule, this will apply the transform ``\\frac{x-\\mu}{\\sigma}``, (where ``x\\sim N(\\mu,\\sigma)``), to each data dimension.
For multivariate standardization, see [`Decorrelator`](@ref) 
"""
zscore_scale() = ElementwiseScaler(ZScoreScaling)

ElementwiseScaler(::Type{UAS}) where {UAS <: UnivariateAffineScaling} =
    ElementwiseScaler{UAS, Vector{Float64}, Vector, Vector, Vector, Vector}(Float64[], Float64[], [], [], [], [])

"""
$(TYPEDSIGNATURES)

Gets the UnivariateAffineScaling type `T`
"""
get_type(es::ElementwiseScaler{T}) where {T} = T

"""
$(TYPEDSIGNATURES)

Gets the `shift` field of the `ElementwiseScaler`
"""
get_shift(es::ElementwiseScaler) = es.shift

"""
$(TYPEDSIGNATURES)

Gets the `scale` field of the `ElementwiseScaler`
"""
get_scale(es::ElementwiseScaler) = es.scale

"""
$(TYPEDSIGNATURES)

returns the `data_encoder_mat` field of the `ElementwiseScaler`.
"""
get_data_encoder_mat(es::ElementwiseScaler) = es.data_encoder_mat

"""
$(TYPEDSIGNATURES)

returns the `data_decoder_mat` field of the `ElementwiseScaler`.
"""
get_data_decoder_mat(es::ElementwiseScaler) = es.data_decoder_mat

"""
$(TYPEDSIGNATURES)

returns the `struct_encoder_mat` field of the `ElementwiseScaler`.
"""
get_struct_encoder_mat(es::ElementwiseScaler) = es.struct_encoder_mat

"""
$(TYPEDSIGNATURES)

returns the `struct_decoder_mat` field of the `ElementwiseScaler`.
"""
get_struct_decoder_mat(es::ElementwiseScaler) = es.struct_decoder_mat

function Base.show(io::IO, es::ElementwiseScaler)
    out = "ElementwiseScaler: $(get_type(es))"
    print(io, out)
end


function initialize_processor!(
    es::ElementwiseScaler,
    data::MM,
    ::Type{QS},
) where {MM <: AbstractMatrix, QS <: QuartileScaling}
    quartiles_vec = [quantile(dd, [0.25, 0.5, 0.75]) for dd in eachrow(data)]
    quartiles_mat = reduce(hcat, quartiles_vec) # 3 rows: Q1, Q2, and Q3

    # we use these more for saving readable data
    append!(get_shift(es), quartiles_mat[2, :])
    append!(get_scale(es), (quartiles_mat[3, :] - quartiles_mat[1, :]))
end

function initialize_processor!(
    es::ElementwiseScaler,
    data::MM,
    ::Type{MMS},
) where {MM <: AbstractMatrix, MMS <: MinMaxScaling}
    minmax_vec = [[minimum(dd), maximum(dd)] for dd in eachrow(data)]
    minmax_mat = reduce(hcat, minmax_vec) # 2 rows: min max
    append!(get_shift(es), minmax_mat[1, :])
    append!(get_scale(es), (minmax_mat[2, :] - minmax_mat[1, :]))
end

function initialize_processor!(
    es::ElementwiseScaler,
    data::MM,
    ::Type{ZSS},
) where {MM <: AbstractMatrix, ZSS <: ZScoreScaling}
    stat_vec = [[mean(dd), std(dd)] for dd in eachrow(data)]
    stat_mat = reduce(hcat, stat_vec) # 2 rows: mean, std
    append!(get_shift(es), stat_mat[1, :])
    append!(get_scale(es), stat_mat[2, :])
end

function initialize_processor!(es::ElementwiseScaler, data::MM) where {MM <: AbstractMatrix}
    if length(get_shift(es)) == 0
        T = get_type(es)
        initialize_processor!(es, data, T)

        # we explicitly make the encoder/decoder maps
        data_encoder_map = LinearMap(
            x -> (x .- get_shift(es)) ./ get_scale(es), # Ax
            x -> (x .- get_shift(es)) ./ get_scale(es), # A'x
            size(data, 1), # size(A,1)
            size(data, 1), # size(A,2)
        )
        data_decoder_map = LinearMap(
            x -> x .* get_scale(es) .+ get_shift(es), # Ax        
            x -> x .* get_scale(es) .+ get_shift(es), # A'x 
            size(data, 1), # size(A,1)
            size(data, 1), # size(A,2)
        )
        # the encoder for the structure matrix does not have a shift
        struct_encoder_map = LinearMap(
            x -> x ./ get_scale(es), # Ax
            x -> x ./ get_scale(es), # A'x
            size(data, 1), # size(A,1)
            size(data, 1), # size(A,2)
        )
        struct_decoder_map = LinearMap(
            x -> x .* get_scale(es), # Ax        
            x -> x .* get_scale(es), # A'x 
            size(data, 1), # size(A,1)
            size(data, 1), # size(A,2)
        )
        push!(get_data_encoder_mat(es), data_encoder_map)
        push!(get_data_decoder_mat(es), data_decoder_map)
        push!(get_struct_encoder_mat(es), struct_encoder_map)
        push!(get_struct_decoder_mat(es), struct_decoder_map)

    end
end

"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` encoder, on a columns-are-data matrix
"""
function encode_data(es::ElementwiseScaler, data::MM) where {MM <: AbstractMatrix}
    out = zeros(size(data))
    enc = get_data_encoder_mat(es)[1]
    mul!(out, enc, data)  # must use this form to get matrix output of enc*out
    return out
end

"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` decoder, on a columns-are-data matrix
"""
function decode_data(es::ElementwiseScaler, data::MM) where {MM <: AbstractMatrix}
    out = zeros(size(data))
    dec = get_data_decoder_mat(es)[1]
    mul!(out, dec, data)  # must use this form to get matrix output of dec*out
    return out
end

"""
$(TYPEDSIGNATURES)

Computes and populates the `shift` and `scale` fields for the `ElementwiseScaler`
"""
initialize_processor!(
    es::ElementwiseScaler,
    data::MM,
    structure_matrices,
    structure_vectors,
) where {MM <: AbstractMatrix} = initialize_processor!(es, data)


"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` encoder to a provided structure matrix. If the structure matrix is a LinearMap, then the encoded structure matrix remains a LinearMap.
"""
function encode_structure_matrix(es::ElementwiseScaler, structure_matrix::SM) where {SM <: StructureMatrix}
    encoder_mat = get_struct_encoder_mat(es)[1]
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` decoder to a provided structure matrix. If the structure matrix is a LinearMap, then the encoded structure matrix remains a LinearMap.
"""
function decode_structure_matrix(es::ElementwiseScaler, enc_structure_matrix::SM) where {SM <: StructureMatrix}
    decoder_mat = get_struct_decoder_mat(es)[1]
    return decoder_mat * enc_structure_matrix * decoder_mat'
end
