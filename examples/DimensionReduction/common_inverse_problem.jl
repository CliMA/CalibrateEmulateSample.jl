using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Statistics
using Distributions

# Inverse problem will be taken from (Cui, Tong, 2021) https://arxiv.org/pdf/2101.02417, example 7.1
include("forward_maps.jl")


function linear_exp_inverse_problem(input_dim, output_dim, rng)
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

    # forward map
    # random linear-exp forward map from Stewart 1980: https://www.jstor.org/stable/2156882?seq=2
    U = qr(randn(rng, (output_dim, output_dim))).Q
    V = qr(randn(rng, (input_dim, input_dim))).Q
    λ0 = 100.0
    β_λ = -1
    Λ = Diagonal([λ0 * (1.0 * j)^β_λ for j in 1:output_dim])
    A = U * Λ * V[1:output_dim, :] # output x input
    model = LinearExp(input_dim, output_dim, A)

    # generate data sample
    obs_noise_std = 1.0
    obs_noise_cov = (obs_noise_std^2) * I(output_dim)
    noise = rand(rng, MvNormal(zeros(output_dim), obs_noise_cov))
    true_parameter = reshape(ones(input_dim), :, 1)
    y = vec(forward_map(true_parameter, model) + noise)
    return prior, y, obs_noise_cov, model, true_parameter
end
