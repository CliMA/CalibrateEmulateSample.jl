# included in Utilities.jl

using Manifolds, Manopt

mutable struct LikelihoodInformed{FT <: Real} <: PairedDataContainerProcessor
    encoder_mat::Union{Nothing, AbstractMatrix}
    decoder_mat::Union{Nothing, AbstractMatrix}
    apply_to::Union{Nothing, AbstractString}
    dim_criterion::Tuple{Symbol, <:Number}
    α::FT
    grad_type::Symbol
    use_prior_samples::Bool
end

function likelihood_informed(retain_KL; alpha = 0.0, grad_type = :localsl, use_prior_samples = true)
    if grad_type ∉ [:linreg, :localsl]
        @error "Unknown grad_type=$grad_type"
    end

    LikelihoodInformed(nothing, nothing, nothing, (:retain_KL, retain_KL), alpha, grad_type, use_prior_samples)
end

get_encoder_mat(li::LikelihoodInformed) = li.encoder_mat
get_decoder_mat(li::LikelihoodInformed) = li.decoder_mat

function initialize_processor!(
    li::LikelihoodInformed,
    in_data::MM,
    out_data::MM,
    ::Dict{Symbol, <:StructureMatrix},
    output_structure_matrices::Dict{Symbol, <:StructureMatrix},
    input_structure_vectors::Dict{Symbol, <:StructureVector},
    output_structure_vectors::Dict{Symbol, <:StructureVector},
    apply_to::AbstractString,
) where {MM <: AbstractMatrix}
    output_dim = size(out_data, 2)

    if isnothing(get_encoder_mat(li))
        α = li.α
        y = if α ≈ 0.0
            # For α=0, it doesn't matter what this value is, so we avoid requiring its presence
            zeros(size(out_data, 1))
        else
            get_structure_vec(output_structure_vectors, :observation)
        end
        samples_in, samples_out = if li.use_prior_samples
            @assert α ≈ 0.0
            (
                get_structure_vec(input_structure_vectors, :prior_samples_in),
                get_structure_vec(output_structure_vectors, :prior_samples_out),
            )
        else
            (in_data, out_data)
        end
        obs_noise_cov = get_structure_mat(output_structure_matrices, :obs_noise_cov)
        noise_cov_inv = inv(obs_noise_cov)

        li.apply_to = apply_to

        grads = if li.grad_type == :linreg
            grad = (samples_out .- mean(samples_out; dims = 2)) / (samples_in .- mean(samples_in; dims = 2))
            fill(grad, size(samples_in, 2))
        else
            @assert li.grad_type == :localsl

            map(eachcol(samples_in)) do u
                # TODO: It might be interesting to introduce a parameter to weight this distance with.
                #       This can be a scalar or a matrix; in the latter case, we can even use the covariance
                #       of the samples (or the prior covariance).
                weights = exp.(-1/2 * norm.(eachcol(u .- samples_in)).^2)
                D = Diagonal(sqrt.(weights))
                uw = (samples_in .- mean(samples_in * Diagonal(weights); dims = 2)) * D
                gw = (samples_out .- mean(samples_out * Diagonal(weights); dims = 2)) * D
                gw / uw
            end
        end

        li.encoder_mat = if apply_to == "in" || α ≈ 0
            decomp = if apply_to == "in"
                eigen(mean(grad' * noise_cov_inv * ((1-α)obs_noise_cov + α^2 * (y - g) * (y - g)') * noise_cov_inv * grad for (g, grad) in zip(eachcol(samples_out), grads)), sortby = (-))
            else
                @assert apply_to == "out"
                eigen(mean(grad * grad' for grad in grads), obs_noise_cov, sortby = (-))
            end

            if li.dim_criterion[1] == :retain_KL
                retain_KL = li.dim_criterion[2]
                sv_cumsum = cumsum(decomp.values) / sum(decomp.values)
                trunc_val = findfirst(x -> (x ≥ retain_KL), sv_cumsum)
            else
                @assert li.dim_criterion[1] == :dimension
                trunc_val = li.dim_criterion[2]
            end
            li.encoder_mat = decomp.vectors[:, 1:trunc_val]'
        else
            @assert apply_to == "out"
            @warn "Using LikelihoodInformed on output data with α≠0 triggers a manifold optimization process that may take some time."

            k = if li.dim_criterion[1] == :retain_KL
                1
            else
                @assert li.dim_criterion[1] == :dimension
                li.dim_criterion[2]
            end
            Vs = nothing
            while true
                M = Grassmann(output_dim, k)

                f = (_, Vs) -> begin
                    prec = noise_cov_inv - Vs * inv(Vs' * obs_noise_cov * Vs) * Vs'
                    tr(mean(
                        grad' * prec * ((1-α)I + α^2 * (y - g)*(y - g)') * prec * grad
                        for (g, grad) in zip(eachcol(out_data), grads)
                    ))
                end
                egrad = (_, Vs) -> begin
                    B = Vs * inv(Vs' * obs_noise_cov * Vs) * Vs'
                    prec = noise_cov_inv - B
                    

                    -2mean(begin
                        A = ((1-α)I + α^2 * (y - g)*(y - g)')
                        S = grad * grad'
                        (I - obs_noise_cov * B) * (S * prec * A + A * prec * S)
                    end for (g, grad) in zip(eachcol(out_data), grads)) * B * Vs
                end
                rgrad = (M, Vs) -> begin
                    (I - Vs*Vs') * egrad(M, Vs)
                end

                Vs = Matrix(qr(randn(output_dim, k)))
                quasi_Newton!(M, f, rgrad, Vs; stopping_criterion = StopWhenGradientNormLess(3.0))

                if li.dim_criterion[1] == :retain_KL
                    retain_KL = li.dim_criterion[2]
                    ref = f(M, zeros(output_dim, 0))
                    if f(M, Vs) / ref ≤ 1 - retain_KL
                        break # TODO: Start bisecting?
                    else
                        k *= 2
                    end
                else
                    @assert li.dim_criterion[1] == :dimension
                    break
                end
            end

            Vs'
        end
        li.decoder_mat = li.encoder_mat'
    end
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` encoder, on a columns-are-data matrix
"""
function encode_data(li::LikelihoodInformed, data::MM) where {MM <: AbstractMatrix}
    encoder_mat = get_encoder_mat(li)
    return encoder_mat * data
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` decoder, on a columns-are-data matrix
"""
function decode_data(li::LikelihoodInformed, data::MM) where {MM <: AbstractMatrix}
    decoder_mat = get_decoder_mat(li)
    return decoder_mat * data
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` encoder to a provided structure matrix
"""
function encode_structure_matrix(
    li::LikelihoodInformed,
    structure_matrix::SM,
) where {SM <: StructureMatrix}
    encoder_mat = get_encoder_mat(li)
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` decoder to a provided structure matrix
"""
function decode_structure_matrix(
    li::LikelihoodInformed,
    structure_matrix::SM,
) where {SM <: StructureMatrix}
    decoder_mat = get_decoder_mat(li)
    return decoder_mat * structure_matrix * decoder_mat'
end
