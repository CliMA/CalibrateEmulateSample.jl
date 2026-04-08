using Distributions
using EnsembleKalmanProcesses
using JLD2
using LinearAlgebra
using Random

include("./models.jl")
# select the forward map
mod_types=["linear"]
mod_type = mod_types[1]
@info "Executing model type $(mod_type)"

datadir = joinpath(@__DIR__,"datafiles")
if !isdir(datadir)
    mkpath(datadir)
end

rng = Random.MersenneTwister(41)
input_dim = 100
output_dim = 100

num_trials = 5
for trial in 1:num_trials
    @info "Trial $trial"
    if mod_type == "linear"
        prior_cov, y, obs_noise_cov, model, true_parameter = lin(input_dim, output_dim, rng)
    else
        bad_model(mod_type, mod_types)
    end
    
    prior_obj = ParameterDistribution(
        Parameterized(MvNormal(zeros(size(prior_cov, 1)), prior_cov)),
        fill(no_constraint(), size(prior_cov, 1)),
        "$(mod_type)_prior",
    )

    n_ensemble = 50
    initial_ensemble = construct_initial_ensemble(rng, prior_obj, n_ensemble)
    ekp = EnsembleKalmanProcess(
        initial_ensemble,
        y,
        obs_noise_cov,
        TransformInversion();
        rng,
    )

    n_iters = 0
    for i in 1:50
        n_iters += 1
        G_ens = hcat([forward_map(param, model) for param in eachcol(get_ϕ_final(prior_obj, ekp))]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            break
        end
    end
    @info "EKP iterations: $n_iters"
    @info "Loss over iterations: $(get_error(ekp))"
    @info "Timesteps: $(ekp.Δt)"
    @info "Checkpoints: $(get_algorithm_time(ekp))"
    αs = vec([0 get_algorithm_time(ekp)[1:end]...])
    ekp_samples = Dict(α => (get_u(ekp, i), get_g(ekp, i)) for (i,α) in enumerate(αs[1:end-1]))

    # evaluate at final time
    G_ens = hcat([forward_map(param, model) for param in eachcol(get_ϕ_final(prior_obj, ekp))]...)
    ekp_samples[αs[end]] = (get_u(ekp,length(αs)), G_ens)
    
    #! format: off
    save(
        joinpath(datadir,"ekp_$(mod_type)_$(trial).jld2"),
        "ekpobj", ekp,
        "ekp_samples", ekp_samples,
        "prior_obj", prior_obj,
        "y", y,
        "obs_noise_cov", obs_noise_cov,
        "model", model,
        "true_parameter", true_parameter,
        "alphas", αs,
    )
    #! format: on
end
