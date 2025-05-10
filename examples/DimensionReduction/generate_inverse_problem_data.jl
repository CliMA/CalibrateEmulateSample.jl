using Plots
using EnsembleKalmanProcesses
using Random
using JLD2

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

input_dim = 500
output_dim = 50

include("common_inverse_problem.jl")

n_trials = 20
@info "solving $(n_trials) inverse problems with different random forward maps"

for trial in 1:n_trials
    prior, y, obs_noise_cov, model, true_parameter = linear_exp_inverse_problem(input_dim, output_dim, rng)

    n_ensemble = 80
    n_iters_max = 20

    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, TransformInversion(); rng = rng)

    n_iters = [0]
    for i in 1:n_iters_max
        params_i = get_ϕ_final(prior, ekp)
        G_ens = hcat([forward_map(params_i[:, i], model) for i in 1:n_ensemble]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            n_iters[1] = i - 1
            break
        end
    end

    @info "Iteration of posterior convergence: $(n_iters[1])"
    @info "Loss over iterations:" get_error(ekp)
    save(
        "ekp_$(trial).jld2",
        "ekp",
        ekp,
        "prior",
        prior,
        "y",
        y,
        "obs_noise_cov",
        obs_noise_cov,
        "model",
        model,
        "true_parameter",
        true_parameter,
    )
end
