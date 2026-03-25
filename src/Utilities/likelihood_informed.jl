# included in Utilities.jl

using Manifolds, Manopt

export LikelihoodInformed, likelihood_informed

"""
$(TYPEDEF)

Uses both input and output data to learn a subspace that allows for a reduced posterior which is close to the full posterior.

Preferred construction is with the [`likelihood_informed`](@ref) method.

# Fields
$(TYPEDFIELDS)
"""
mutable struct LikelihoodInformed{VV1<:AbstractVector, VV2<:AbstractVector, VV3<:AbstractVector, VV4<:AbstractVector, FT <: Real} <: PairedDataContainerProcessor
    encoder_mat::VV1
    decoder_mat::VV2
    data_mean::VV3
    retain_kl::FT
    apply_to::Union{Nothing, AbstractString}
    iters::VV4
    grad_type::Symbol
    use_data_as_samples::Bool
end

"""
$(TYPEDSIGNATURES)

Constructs the `LikelihoodInformed` struct. Keywords:
- `retain_kl`: the method will attempt to limit the KL divergence of the true posterior from the reduced posterior to a value proportional to (1 - retain_kl). Choose `retain_kl` close to 1 to get a good approximation in a large subspace, and reduce it to get a worse approximation in a smaller subspace.
- `iters`[=[1]]: the likelihood-informed data processor requires samples from the distribution ∝ π_prior(x) π_likelihood(y | x)^α with α ∈ [0, 1]. Here, `iter` indicates the structure vector iterations to use, as sampled from these distributions. For how to pass in these samples, see the `use_data_as_samples` parameter.
- `grad_type`[=:localsl]: how the gradient of the forward model at the samples will be approximated. Choose from `:linreg` (global linear regression) and `:localsl` (localized statistical linearization; see [Wacker, 2025]).
- `use_data_as_samples`[=false]: if this parameter is `true`, then the data being processed (the training data for the emulator) will be used as the samples mentioned earlier. This means that they must be from the correct distribution corresponding to the chosen `alpha`. If this parameter is `false`, then the method expects `:samples_in` and `:samples_out` structure vectors that contain the samples instead.
"""
function likelihood_informed(; retain_kl = 1, iters=1, grad_type = :linreg, use_data_as_samples = false)
    if grad_type ∉ [:linreg, :localsl]
        @error "Unknown grad_type=$grad_type"
    end
    if !isa(iters, AbstractVector)
        iters=[iters]
    end
    if !(eltype(iters) <: Integer)
      throw(ArgumentError, "Iterations must be passed as an Int or Vec{Int}. This corresponds to which of the structure-vectors are used to construct the subspace")
    end
    
    LikelihoodInformed([], [], [], retain_kl, nothing, iters, grad_type, use_data_as_samples)
end

get_encoder_mat(li::LikelihoodInformed) = li.encoder_mat
get_decoder_mat(li::LikelihoodInformed) = li.decoder_mat
get_data_mean(li::LikelihoodInformed) = li.data_mean
get_retain_kl(li::LikelihoodInformed) = li.retain_kl
get_iters(li::LikelihoodInformed) = li.iters
get_grad_type(li::LikelihoodInformed) = li.grad_type

function Base.show(io::IO, li::LikelihoodInformed)
    out = "LikelihoodInformed"
    out *= ": iters=$(get_iters(li)), grad_type=$(get_grad_type(li))"
    if get_retain_kl(li) < 1.0
        out *= ", retain_kl=$(get_retain_kl(li))"
    end
    print(io, out)
end

function initialize_processor!(
    li::LikelihoodInformed,
    in_data::MM,
    out_data::MM,
    ::Dict{Symbol, SM1},
    output_structure_matrices::Dict{Symbol, SM2},
    input_structure_vectors::Dict{Symbol, SV1},
    output_structure_vectors::Dict{Symbol, SV2},
    apply_to::AS,
) where {MM <: AbstractMatrix, SM1 <:StructureMatrix, SM2 <: StructureMatrix, SV1 <: StructureVector, SV2 <: StructureVector, AS <: AbstractString}
    input_dim = size(in_data, 1)
    output_dim = size(out_data, 1)

    
    if length(get_encoder_mat(li))==0
        iters = get_iters(li)      
        alphas = get_structure_vec(input_structure_vectors, :dt)
        @info "Constructing a likelihood-informed subspace using, \n iterations:$(get_iters(li)), \n α: $(alphas[iters]) "        
        diagnostic_mats = Dict{Int64, AbstractMatrix}()
        samples_means = Dict{Int64, AbstractMatrix}()
        diagnostic_fs = []
        diagnostic_egrads = []

        obs_noise_cov = Matrix(get_structure_mat(output_structure_matrices, :obs_noise_cov))
        # We convert this to a matrix here to avoid dealing with LinearMaps.jl 
        obs_whitened = if obs_noise_cov ≈ I
            obs_noise_cov = I(output_dim)
            true
        else
            @warn "Consider using decorrelate_structure_mat to gain obs_noise_cov = I before calling likelihood_informed"
            false
        end
        noise_cov_inv = inv(obs_noise_cov)
        
        li.apply_to = apply_to
        
        for (it, α) in zip(iters, alphas[iters]) # take the iterations from alpha
            #(NB! "it" may not be 1:end)

            # construct the diagnostic matrix, for which we take the eigendecomposition to find encoder/decoder matrices
            y = if α ≈ 0.0
                # For α=0, it doesn't matter what this value is, so we avoid requiring its presence
                zeros(size(out_data, 1))
            else
                vec(get_structure_vec(output_structure_vectors, :observation)[1])
            end
            
            # take samples from the appropriate distribution as prescribed by alpha
            samples_in, samples_out = if li.use_data_as_samples
                (in_data, out_data)
            else
                (
                    get_structure_vec(input_structure_vectors, :samples_in)[it],
                    get_structure_vec(output_structure_vectors, :samples_out)[it],
                )
            end
            samples_in_mean = mean(samples_in,dims=2)
            samples_out_mean = mean(samples_out,dims=2)
            
            grads = if get_grad_type(li) == :linreg
                grad = (samples_out .- samples_out_mean) / (samples_in .- samples_in_mean)
                fill(grad, size(samples_in, 2))
            else
                @assert get_grad_type(li) == :localsl
                
                map(eachcol(samples_in)) do u
                    # TODO: It might be interesting to introduce a parameter to weight this distance with.
                    #       This can be a scalar or a matrix; in the latter case, we can even use the covariance
                    #       of the samples (or the prior covariance).
                    weights = exp.(-1 / 2 * norm.(eachcol(u .- samples_in)) .^ 2)
                    weights ./= sum(weights)
                    D = Diagonal(sqrt.(weights))
                    uw = (samples_in .- sum(samples_in * Diagonal(weights); dims = 2)) * D
                    gw = (samples_out .- sum(samples_out * Diagonal(weights); dims = 2)) * D
                    gw / uw
                end
            end

            # get the mean shift
            samples_means[it] = if apply_to == "in" 
                samples_in_mean
            else
                samples_out_mean
            end


            # get the diagnostics:
            # either we get the mats to be truncated,
            # or we find the functions & gradients when matrix-free
            #            if apply_to == "in" || (α ≈ 0 && obs_whitened) #if we keep this branch then we have to possibly combine diagnostic mats and f's
            if apply_to == "in" 
                diagnostic_mats[it] = hermitianpart( 
                    mean(
                        grad' *
                            noise_cov_inv *
                            ((1 - α)*obs_noise_cov + α^2 * (y - g) * (y - g)') *
                            noise_cov_inv *
                            grad for (g, grad) in zip(eachcol(samples_out), grads)
                                ),
                )
            #elseif apply_to == "out" && (α ≈ 0 && obs_whitened)
                # if we keep this branch then we get an annoying cross-problem where we have to combine both diagnostic mats and diagnostic f's. So we can make them both "f's" for now
            #    diagnostic_mats[it] = hermitianpart(mean(grad * grad' for grad in grads))
            else
                #                @assert apply_to == "out" &&  !(α ≈ 0 && obs_whitened)
                   @assert apply_to == "out"
                # Need to represent the "f" and "egrad" functions for this α
                f =
                    (_, Vs) -> begin
                        prec = noise_cov_inv - Vs * inv(Vs' * obs_noise_cov * Vs) * Vs'
                        tr(
                            mean(
                                grad' * prec * ((1 - α)obs_noise_cov + α^2 * (y - g) * (y - g)') * prec * grad
                                for (g, grad) in zip(eachcol(out_data), grads)
                                    ),
                        )
                    end
                egrad =
                    (_, Vs) -> begin
                        B = Vs * inv(Vs' * obs_noise_cov * Vs) * Vs'
                        prec = noise_cov_inv - B
                        
                        -2obs_noise_cov * prec * mean(
                            begin
                                A = ((1 - α)obs_noise_cov + α^2 * (y - g) * (y - g)')
                                S = grad * grad'
                                (S * prec * A + A * prec * S)
                            end for (g, grad) in zip(eachcol(out_data), grads)
                                ) *
                                    B *
                                    Vs
                    end
                push!(diagnostic_fs, f)
                push!(diagnostic_egrads, egrad)                
            end
           
        end
        
        # summarize path of diagnostic matrices, if we have more than one
        encoder_mat=nothing
        if length(keys(diagnostic_mats))>0 # using diagnostic_mats
            if length(iters)>1
                # trap rule
                alpha_weight = zeros(length(iters))
                Δa=diff(alphas[iters])
                alpha_weight[1:end-1] .+= Δa ./ 2  
                alpha_weight[2:end] .+= Δa ./ 2  
                alpha_weight ./= sum(alpha_weight)
                diagnostic_mat = sum(alpha_weight[i]*diagnostic_mats[iter] for (i,iter) in enumerate(iters[2:end]))
                samples_mean = sum(alpha_weight[i]*samples_means[iter] for (i,iter) in enumerate(iters[2:end]))
            else
                diagnostic_mat = diagnostic_mats[iters[1]]
                samples_mean = samples_means[iters[1]]
            end
            
            # then get the eigen-decomposition
            decomp = eigen(diagnostic_mat, sortby = (-))
            sv_cumsum = cumsum(decomp.values) / sum(decomp.values)
            retain_kl = get_retain_kl(li)
            if retain_kl >= 1.0
                trunc_val = apply_to == "in" ? input_dim : output_dim
                
            else                
                trunc_val = findfirst(x -> (x ≥ retain_kl), sv_cumsum)
                trunc_val = isnothing(trunc_val) ? (apply_to == "in" ? input_dim : output_dim) : trunc_val
                @info "    truncating at $trunc_val/$(length(sv_cumsum)) retaining $(100.0*sv_cumsum[trunc_val])% of the KL divergence reduction"
            end
            encoder_mat = decomp.vectors[:, 1:trunc_val]' 
        else # using diagnostic_f's and diagnostic_egrads
            @assert length(diagnostic_fs)>0 && length(diagnostic_egrads)>0 
            @assert apply_to == "out"
            
            diagnostic_f, diagnostic_egrad, samples_mean =
                if length(iters)>1
                    method="trap_rule"
                    if method == "trap_rule"
                        
                        alpha_weight = zeros(length(iters))
                        Δa=diff(alphas[iters])
                        alpha_weight[1:end-1] .+= Δa ./ 2  
                        alpha_weight[2:end] .+= Δa ./ 2  
                        alpha_weight ./= sum(alpha_weight)
                        diagnostic_f = (x,Vs) -> sum(w * f(x,Vs) for (f, w) in zip(diagnostic_fs, alpha_weight))
                        diagnostic_egrad = (x,Vs) -> sum(w * egrad(x,Vs) for (egrad, w) in zip(diagnostic_egrads, alpha_weight))
                        samples_mean = sum(alpha_weight[i]*samples_means[iter] for (i,iter) in enumerate(iters[2:end]))
                        
                        diagnostic_f, diagnostic_egrad, samples_mean
                    elseif method == "final"
                        diagnostic_f = diagnostic_fs[end]
                        diagnostic_egrad = diagnostic_egrads[end]
                        samples_mean = samples_means[iters[end]]

                        diagnostic_f, diagnostic_egrad, samples_mean
                    end
                else
                    diagnostic_f = diagnostic_fs[1]
                    diagnostic_egrad = diagnostic_egrads[1]
                    samples_mean = samples_means[iters[1]]
                    
                    diagnostic_f, diagnostic_egrad, samples_mean
                end
            
            @warn "Using LikelihoodInformed on output data with α≠0 or with obs_noise_cov≠I triggers a manifold optimization process that may take some time. If α=0, consider using decorrelate_structure_mat to gain obs_noise_cov = I before calling likelihood_informed"
            
            k = 1
            Vs = nothing
            retain_kl = get_retain_kl(li)
            while true
                M = Grassmann(output_dim, k)
                
                diagnostic_rgrad = (M, Vs) -> begin
                    (I - Vs * Vs') * diagnostic_egrad(M, Vs)
                end
                
                Vs = Matrix(qr(randn(output_dim, k)).Q)
                quasi_Newton!(M, diagnostic_f, diagnostic_rgrad, Vs; stopping_criterion = StopWhenCostChangeLess(0.1))
                
                
                ref = diagnostic_f(M, zeros(output_dim, 0))
                val = diagnostic_f(M, Vs)
                if val / ref ≤ 1 - retain_kl
                    @info "    truncating at $k/$output_dim retaining $(100.0*(1-val/ref))% of the KL divergence reduction"
                    break # TODO: Start bisecting?
                else
                    newk = min(2k, output_dim)
                    @info "      increasing k from $k to $newk"
                    k = newk
                end
                    
            end
            
            encoder_mat = Vs' # setting encoder mat
           
        end            
        decoder_mat = encoder_mat'

        # creat the linear maps:
        # we explicitly make the encoder/decoder maps 
        encoder_map = LinearMap(
            x -> encoder_mat * x, # Ax
            x -> encoder_mat' * x, # A'x
            size(encoder_mat, 1), # size(A,1)
            size(encoder_mat, 2), # size(A,2)
        )

        decoder_map = LinearMap(
            x -> decoder_mat * x, # Ax
            x -> decoder_mat' * x, # A'x
            size(decoder_mat, 1), # size(A,1)
            size(decoder_mat, 2), # size(A,2)
        )

        push!(get_encoder_mat(li), encoder_map)
        push!(get_decoder_mat(li), decoder_map)

        push!(get_data_mean(li), vec(samples_mean))
        
    end
    
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` encoder, on a columns-are-data matrix
"""
function encode_data(li::LikelihoodInformed, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(li)[1]
    encoder_mat = get_encoder_mat(li)[1]
    out = zeros(size(encoder_mat, 1), size(data, 2))
    mul!(out, encoder_mat, data .- data_mean)  
    return out
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` decoder, on a columns-are-data matrix
"""
function decode_data(li::LikelihoodInformed, data::MM) where {MM <: AbstractMatrix}
    data_mean = get_data_mean(li)[1]
    decoder_mat = get_decoder_mat(li)[1]
    out = zeros(size(decoder_mat, 1), size(data, 2))    
    mul!(out, decoder_mat, data)  # must use this form to get matrix output of dec*out
    return out .+ data_mean
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` encoder to a provided structure matrix
"""
function encode_structure_matrix(li::LikelihoodInformed, structure_matrix::SM) where {SM <: StructureMatrix}
    encoder_mat = get_encoder_mat(li)[1]
    return encoder_mat * structure_matrix * encoder_mat'
end

"""
$(TYPEDSIGNATURES)

Apply the `LikelihoodInformed` decoder to a provided structure matrix
"""
function decode_structure_matrix(li::LikelihoodInformed, structure_matrix::SM) where {SM <: StructureMatrix}
    decoder_mat = get_decoder_mat(li)[1]
    return decoder_mat * structure_matrix * decoder_mat'
end
