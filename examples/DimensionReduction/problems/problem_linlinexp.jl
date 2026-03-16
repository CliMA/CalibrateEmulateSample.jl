using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Statistics
using Distributions

include("forward_maps.jl")

function linlinexp(input_dim, output_dim, rng)
    # prior
    γ0 = 4.0
    β_γ = -2
    Γ = Diagonal([γ0 * (1.0 * j)^β_γ for j in 1:input_dim])
    prior_dist = MvNormal(zeros(input_dim), Γ)
    prior = ParameterDistribution(
        Dict(
            "distribution" => Parameterized(prior_dist),
            "constraint" => repeat([no_constraint()], input_dim),
            "name" => "param_$(input_dim)",
        ),
    )

    U = qr(randn(rng, (output_dim, output_dim))).Q
    V = qr(randn(rng, (input_dim, input_dim))).Q
    λ0 = 100.0
    β_λ = -1
    Λ = Diagonal([λ0 * (1.0 * j)^β_λ for j in 1:output_dim])
    A = U * Λ * V[1:output_dim, :] # output x input
    model = LinLinExp(input_dim, output_dim, A)

    # generate data sample
    obs_noise_cov = Diagonal([Float64(j)^(-1 / 2) for j in 1:output_dim])
    noise = rand(rng, MvNormal(zeros(output_dim), obs_noise_cov))
    # true_parameter = reshape(ones(input_dim), :, 1)
    true_parameter = rand(prior_dist)
    y = vec(forward_map(true_parameter, model) + noise)
    return prior, y, obs_noise_cov, model, true_parameter
end

struct LinLinExp{AM <: AbstractMatrix} <: ForwardMapType
    input_dim::Int
    output_dim::Int
    G::AM
end

function forward_map(X::AVorM, model::LinLinExp) where {AVorM <: AbstractVecOrMat}
    return model.G * (X .* exp.(0.05X))
end

function jac_forward_map(X::AbstractVector, model::LinLinExp)
    return model.G * Diagonal(exp.(0.05X) .* (1 .+ 0.05X))
end

function jac_forward_map(X::AbstractMatrix, model::LinLinExp)
    return [jac_forward_map(x, model) for x in eachcol(X)]
end
