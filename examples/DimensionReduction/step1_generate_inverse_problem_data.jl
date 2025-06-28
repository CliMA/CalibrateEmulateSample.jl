using AdvancedMH
using Distributions
using EnsembleKalmanProcesses
using ForwardDiff
using JLD2
using LinearAlgebra
using MCMCChains
using Random

include("./problems/problem_linear.jl")
include("./problems/problem_linear_exp.jl")
include("./problems/problem_lorenz.jl")
include("./problems/problem_linlinexp.jl")

include("./settings.jl")
include("./util.jl")
rng = Random.MersenneTwister(rng_seed)
problem_fun = if problem == "linear"
    linear
elseif problem == "linear_exp"
    linear_exp
elseif problem == "lorenz"
    lorenz
elseif problem == "linlinexp"
    linlinexp
else
    throw("Unknown problem=$problem")
end


for trial in 1:num_trials
    prior, y, obs_noise_cov, model, true_parameter = problem_fun(input_dim, output_dim, rng)

    # [1] EKP run
    n_ensemble = step1_eki_ensemble_size
    n_iters_max = step1_eki_max_iters

    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    ekp = EnsembleKalmanProcess(
        initial_ensemble,
        y,
        obs_noise_cov,
        TransformInversion();
        rng,
        scheduler = EKSStableScheduler(2.0, 0.01),
    )

    n_iters = n_iters_max
    for i in 1:n_iters_max
        G_ens = hcat([forward_map(param, model) for param in eachcol(get_ϕ_final(prior, ekp))]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            n_iters = i - 1
            break
        end
    end
    @info "EKP iterations: $n_iters"
    @info "Loss over iterations: $(get_error(ekp))"

    # [2] MCMC run
    prior_cov, prior_inv, obs_inv = cov(prior), inv(cov(prior)), inv(obs_noise_cov)
    mcmc_samples = zeros(input_dim, 0)
    do_mcmc(
        input_dim,
        x -> begin
            g = forward_map(x, model)
            (-2 \ x' * prior_inv * x - 2 \ (y - g)' * obs_inv * (y - g)) / step1_mcmc_temperature
        end,
        step1_mcmc_num_chains,
        step1_mcmc_samples_per_chain,
        step1_mcmc_sampler,
        prior_cov,
        true_parameter;
        subsample_rate = step1_mcmc_subsample_rate,
    ) do samp, _
        mcmc_samples = hcat(mcmc_samples, samp)
    end
    @info "MCMC finished"

    # [3] Save everything to a file
    #! format: off
    save(
        "datafiles/ekp_$(problem)_$(trial).jld2",
        "ekp", ekp,
        "mcmc_samples", mcmc_samples,
        "prior", prior,
        "y", y,
        "obs_noise_cov", obs_noise_cov,
        "model", model,
        "true_parameter", true_parameter,
    )
    #! format: on
end
