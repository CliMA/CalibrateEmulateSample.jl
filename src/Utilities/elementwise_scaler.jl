# included in Utilities.jl

export UnivariateAffineScaling, ElementwiseScaler, QuartileScaling, MinMaxScaling, ZScoreScaling
export quartile_scale, minmax_scale, zscore_scale
export get_type, get_shift, get_scale

"""
$(TYPEDEF)

The ElementwiseScaler{T} will create an encoding of the data_container via elementwise affine transformations.

Different methods `T` will build different transformations:
- [`quartile_scale`](@ref) : creates `QuartileScaling`,
- [`minmax_scale`](@ref) : creates `MinMaxScaling`
- [`zscore_scale`](@ref) : creates `ZScoreScaling`

and are accessed with [`get_type`](@ref)
"""
struct ElementwiseScaler{T, VV <: AbstractVector} <: DataContainerProcessor
    "storage for the shift applied to data"
    shift::VV
    "storage for the scaling"
    scale::VV
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
    ElementwiseScaler{UAS, Vector{Float64}}(Float64[], Float64[])

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
    end
end

"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` encoder, on a columns-are-data matrix
"""
function encode_data(es::ElementwiseScaler, data::MM) where {MM <: AbstractMatrix}
    out = deepcopy(data)
    for i in 1:size(out, 1)
        out[i, :] .-= get_shift(es)[i]
        out[i, :] /= get_scale(es)[i]
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` decoder, on a columns-are-data matrix
"""
function decode_data(es::ElementwiseScaler, data::MM) where {MM <: AbstractMatrix}
    out = deepcopy(data)
    for i in 1:size(out, 1)
        out[i, :] *= get_scale(es)[i]
        out[i, :] .+= get_shift(es)[i]
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Computes and populates the `shift` and `scale` fields for the `ElementwiseScaler`
"""
initialize_processor!(es::ElementwiseScaler, data::MM, structure_matrix) where {MM <: AbstractMatrix} =
    initialize_processor!(es, data)


"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` encoder to a provided structure matrix
"""
function encode_structure_matrix(
    es::ElementwiseScaler,
    structure_matrix::USorM,
) where {USorM <: Union{UniformScaling, AbstractMatrix}}
    return Diagonal(1 ./ get_scale(es)) * structure_matrix * Diagonal(1 ./ get_scale(es))
end

"""
$(TYPEDSIGNATURES)

Apply the `ElementwiseScaler` decoder to a provided structure matrix
"""
function decode_structure_matrix(
    es::ElementwiseScaler,
    enc_structure_matrix::USorM,
) where {USorM <: Union{UniformScaling, AbstractMatrix}}
    return Diagonal(get_scale(es)) * enc_structure_matrix * Diagonal(get_scale(es))
end
