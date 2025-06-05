using EnsembleKalmanProcesses
using Random
using JLD2

include("./problems/problem_linear_exp.jl")
include("./problems/problem_lorenz.jl")

include("./settings.jl")
rng = Random.MersenneTwister(rng_seed)
problem_fun = if problem == "lorenz"
    lorenz
elseif problem == "linear_exp"
    linear_exp
else
    throw("Unknown problem=$problem")
end


for trial in 1:num_trials
    prior, y, obs_noise_cov, model, true_parameter = problem_fun(input_dim, output_dim, rng)

    n_ensemble = step1_eki_ensemble_size
    n_iters_max = step1_eki_max_iters

    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, TransformInversion(); rng, scheduler = EKSStableScheduler(2.0, 0.01))

    n_iters = n_iters_max
    for i in 1:n_iters_max
        params_i = get_ϕ_final(prior, ekp)
        G_ens = hcat([forward_map(params_i[:, i], model) for i in 1:n_ensemble]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            n_iters = i - 1
            break
        end
    end

    @info "Iteration of posterior convergence: $n_iters"
    @info "Loss over iterations: $(get_error(ekp))"
    save(
        "data/ekp_$(problem)_$(trial).jld2",
        "ekp", ekp,
        "prior", prior,
        "y", y,
        "obs_noise_cov", obs_noise_cov,
        "model", model,
        "true_parameter", true_parameter,
    )
end

