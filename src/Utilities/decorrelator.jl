# included in Utilities.jl

export Decorrelator, decorrelate_sample_cov, decorrelate_structure_mat, decorrelate
export get_data_mean,
    get_encoder_mat,
    get_decoder_mat,
    get_retain_var,
    get_decorrelate_with,
    get_n_totvar_samples,
    get_max_rank,
    get_psvd_kwargs

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

Depending on the size of the matrix, we perform different options of SVD:

Small Matrix (dim < 3000):
    use LinearAlgebra.svd(Matrix)
Large Matrix (dim > 3000):
    if retain_var = 1.0
        use LowRankApprox.psvd(LinearMap; psvd_kwargs...)
    if retain_var < 1.0
        use TSVD.tsvd(LinearMap)


# Fields
$(TYPEDFIELDS)
"""
struct Decorrelator{VV1, VV2, VV3, FT, NT <: NamedTuple, AS <: AbstractString} <: DataContainerProcessor
    "storage for the data mean"
    data_mean::VV1
    "the matrix used to perform encoding"
    encoder_mat::VV2
    "the inverse of the the matrix used to perform encoding"
    decoder_mat::VV3
    "the fraction of variance to be retained after truncating singular values (1 implies no truncation)"
    retain_var::FT
    "when retain_var < 1, number of samples to estimate the total variance. Larger values reduce the error in approximation at the cost of additional matrix-vector products."
    n_totvar_samples::Int
    "maximum dimension of subspace for `retain_var < 1`. The search may become expensive at large ranks, and therefore can be cut-off in this way"
    max_rank::Int
    "when `retain_var = 1`, the `psvd` algorithm from `LowRankApprox.jl` is used to decorrelate the space. here, kwargs can be passed in as a NamedTuple"
    psvd_kwargs::NT
    "Switch to choose what form of matrix to use to decorrelate the data"
    decorrelate_with::AS
    "When given, use the structure matrix by this name if `decorrelate_with` uses structure matrices. When `nothing`, try to use the only present structure matrix instead."
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
decorrelate(;
    retain_var::FT = Float64(1.0),
    decorrelate_with = "combined",
    structure_mat_name = nothing,
    n_totvar_samples::Int = 500,
    max_rank::Int = 100,
    psvd_kwargs = (; rtol = 1e-5),
) where {FT} = Decorrelator(
    [],
    [],
    [],
    clamp(retain_var, FT(0), FT(1)),
    n_totvar_samples,
    max_rank,
    psvd_kwargs,
    decorrelate_with,
    structure_mat_name,
)

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelator` struct, setting decorrelate_with = "sample_cov". Encoding data with this will ensure that the distribution of data samples after encoding will be `Normal(0,I)`. One can additionally add keywords:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
"""
decorrelate_sample_cov(;
    retain_var::FT = Float64(1.0),
    n_totvar_samples::Int = 500,
    max_rank::Int = 100,
    psvd_kwargs = (; rtol = 1e-5),
) where {FT} = Decorrelator(
    [],
    [],
    [],
    clamp(retain_var, FT(0), FT(1)),
    n_totvar_samples,
    max_rank,
    psvd_kwargs,
    "sample_cov",
    nothing,
)

"""
$(TYPEDSIGNATURES)

Constructs the `Decorrelator` struct, setting decorrelate_with = "structure_mat". This encoding will transform a provided structure matrix into `I`. One can additionally add keywords:
- `retain_var`[=`1.0`]: to project onto the leading singular vectors such that `retain_var` variance is retained
"""
decorrelate_structure_mat(;
    retain_var::FT = Float64(1.0),
    structure_mat_name = nothing,
    n_totvar_samples::Int = 500,
    max_rank::Int = 100,
    psvd_kwargs = (; rtol = 1e-3),
) where {FT} = Decorrelator(
    [],
    [],
    [],
    clamp(retain_var, FT(0), FT(1)),
    n_totvar_samples,
    max_rank,
    psvd_kwargs,
    "structure_mat",
    structure_mat_name,
)

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

returns the `n_totvar_samples` field of the `Decorrelator`.
"""
get_n_totvar_samples(dd::Decorrelator) = dd.n_totvar_samples

"""
$(TYPEDSIGNATURES)

returns the `max_rank` field of the `Decorrelator`.
"""
get_max_rank(dd::Decorrelator) = dd.max_rank

"""
$(TYPEDSIGNATURES)

returns the `psvd_kwargs` field of the `Decorrelator`.
"""
get_psvd_kwargs(dd::Decorrelator) = dd.psvd_kwargs

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
    structure_matrices::Dict{Symbol, SM},
    ::Dict{Symbol, SV},
) where {MM <: AbstractMatrix, SM <: StructureMatrix, SV <: StructureVector}
    if length(get_data_mean(dd)) == 0
        push!(get_data_mean(dd), vec(mean(data, dims = 2)))
    end

    if length(get_encoder_mat(dd)) == 0

        # To unify the actions here, we use LinearMaps.jl
        # Note: can only multiply Linear Maps with vectors, not matrices...
        decorrelate_with = get_decorrelate_with(dd)
        if decorrelate_with == "structure_mat"
            decorrelation_map = get_structure_mat(structure_matrices, dd.structure_mat_name)

        elseif decorrelate_with == "sample_cov"
            decorrelation_action = tsvd_cov_from_samples(data)
            decorrelation_map = create_compact_linear_map(decorrelation_action)

        elseif decorrelate_with == "combined"
            map1 = get_structure_mat(structure_matrices, dd.structure_mat_name)
            decorrelation_action2 = tsvd_cov_from_samples(data)
            map2 = create_compact_linear_map(decorrelation_action2)
            decorrelation_map = map1 + map2 # x -> dm.maps[1].f(x) + dm.maps[2].f(x)
        else
            throw(
                ArgumentError(
                    "Keyword `decorrelate_with` must be taken from [\"sample_cov\", \"structure_mat\", \"combined\"]. Received $(decorrelate_with)",
                ),
            )
        end

        n_totvar_samples = get_n_totvar_samples(dd)
        max_rank = get_max_rank(dd)
        psvd_kwargs = get_psvd_kwargs(dd)
        ret_var = get_retain_var(dd)

        S = []
        Vt = []

        # Three methods of performing: svd, partial svd or truncated svd.
        # svd: convert to matrix and perform actual SVD. [good for small matrix (d < ~3000)]
        # psvd: approximates SVD to a given error tolerance (can truncate too) [good for full svd approx]
        # tsvd: truncates to dimension where resulting SVD has the same s.vals as the true [good of low rank approx]
        max_svd_size = 3000
        if max(size(decorrelation_map)...) < max_svd_size
            # svd options
            svdA = svd(Matrix(decorrelation_map))
            rk = rank(Matrix(decorrelation_map))

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

            push!(S, svdA.S[1:trunc_val])
            push!(Vt, svdA.Vt[1:trunc_val, :])

        else
            if ret_var < 1.0
                # tsvd option
                # approximate Frobenius norm of the map
                norm_approx = zeros(10)
                for i in 1:10
                    samples = randn(size(decorrelation_map, 1), n_totvar_samples)
                    norm_approx[i] =
                        1 / n_totvar_samples * sum([sum(abs2, (decorrelation_map * x)) for x in eachcol(samples)])
                end
                @info "relative error of total variance $(std(norm_approx)/mean(norm_approx))"
                retain_var_max = 1 - std(norm_approx) / mean(norm_approx)

                ret_var_tmp = min(ret_var, retain_var_max)

                for rk in 1:max_rank
                    _, Stmp, Vtmp = tsvd(decorrelation_map, rk)
                    retain_var_current = sum(Stmp .^ 2) / (retain_var_max * mean(norm_approx))
                    if retain_var_current > ret_var_tmp
                        push!(S, Stmp)
                        push!(Vt, Vtmp')
                        @info "    truncating at $(rk)/$(size(data,1)) retaining $(retain_var_current*100)% (+/-$(100*std(norm_approx)/mean(norm_approx)))% of the variance of the structure matrix"
                        break
                    end
                    if rk == max_rank
                        @warn "Maximum tolerance of SVD truncation hit ($(rank_max)), consider increasing `rank_max` in decorrelator, this may also result from `retain_var` being close to 1, while `n_totvar_samples` is low. Proceeding with final iterant."
                        push!(S, Stmp)
                        push!(Vt, Vtmp')

                    end

                end
            else
                # psvd option
                # check V or Vt
                _, Stmp, Vtmp = psvd(decorrelation_map; psvd_kwargs...) # randomized algorithm to avoid building the matrix, but this will be big.

                if length(Stmp) < size(data, 1)
                    @info "    truncating at $(length(Stmp))/$(size(data,1)), as low-rank data detected"
                end
                push!(S, Stmp)
                push!(Vt, Vtmp')

            end
        end



        # Enforce a sign convention for the singular vectors (rows Vt have positive first entry)
        for (j, row) in enumerate(eachrow(Vt[1]))
            if row[1] < 0
                Vt[1][j, :] *= -1
            end
        end

        # we explicitly make the encoder/decoder maps 
        encoder_map = LinearMap(
            x -> Diagonal(1.0 ./ sqrt.(S[1])) * Vt[1] * x, # Ax
            x -> Vt[1]' * Diagonal(1.0 ./ sqrt.(S[1])) * x, # A'x
            size(Vt[1], 1), # size(A,1)
            size(Vt[1], 2), # size(A,2)
        )

        decoder_map = LinearMap(
            x -> Vt[1]' * Diagonal(sqrt.(S[1])) * x, # Ax
            x -> Diagonal(sqrt.(S[1])) * Vt[1] * x, # A'x
            size(Vt[1]', 1), # size(A,1)
            size(Vt[1]', 2), # size(A,2)
        )

        push!(get_encoder_mat(dd), encoder_map)
        push!(get_decoder_mat(dd), decoder_map)
    end
end


"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` encoder, on a columns-are-data matrix
"""
function encode_data(dd::Decorrelator, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(dd)[1]
    encoder_mat = get_encoder_mat(dd)[1]
    out = zeros(size(encoder_mat, 1), size(data, 2))
    mul!(out, encoder_mat, deepcopy(data) .- data_mean)
    return out
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` decoder, on a columns-are-data matrix
"""
function decode_data(dd::Decorrelator, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(dd)[1]
    decoder_mat = get_decoder_mat(dd)[1]
    out = zeros(size(decoder_mat, 1), size(data, 2))
    mul!(out, decoder_mat, deepcopy(data))
    return out .+ data_mean
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` encoder to a provided structure matrix. If the structure matrix is a LinearMap, then the encoded structure matrix remains a LinearMap.
"""
function encode_structure_matrix(dd::Decorrelator, structure_matrix::SM) where {SM <: StructureMatrix}
    encoder_mat = get_encoder_mat(dd)[1]
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `Decorrelator` decoder to a provided structure matrix. If the structure matrix is a LinearMap, then the encoded structure matrix remains a LinearMap.
"""
function decode_structure_matrix(dd::Decorrelator, enc_structure_matrix::SM) where {SM <: StructureMatrix}
    decoder_mat = get_decoder_mat(dd)[1]
    return decoder_mat * enc_structure_matrix * decoder_mat'
end
