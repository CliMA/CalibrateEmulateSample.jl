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

mutable struct CheckpointScheduler <: EnsembleKalmanProcesses.LearningRateScheduler
    αs::Vector{Float64}
    scheduler

    current_index

    CheckpointScheduler(αs, scheduler) = new(αs, scheduler, 2)
end

function EnsembleKalmanProcesses.calculate_timestep!(ekp, g, Δt_new, scheduler::CheckpointScheduler)
    EnsembleKalmanProcesses.calculate_timestep!(ekp, g, Δt_new, scheduler.scheduler)
    if scheduler.current_index <= length(scheduler.αs) && get_algorithm_time(ekp)[end] > scheduler.αs[scheduler.current_index]
        get_algorithm_time(ekp)[end] = scheduler.αs[scheduler.current_index]
        scheduler.current_index += 1
    end

    nothing
end



for trial in 1:num_trials
    prior, y, obs_noise_cov, model, true_parameter = problem_fun(input_dim, output_dim, rng)

    # [1] EKP run
    n_ensemble = step1_eki_ensemble_size

    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    ekp = EnsembleKalmanProcess(
        initial_ensemble,
        y,
        obs_noise_cov,
        TransformInversion();
        rng,
        scheduler = CheckpointScheduler(αs, EKSStableScheduler(2.0, 0.01)),
    )

    n_iters = 0
    while vcat([0.0], get_algorithm_time(ekp))[end] < maximum(αs)
        n_iters += 1
        G_ens = hcat([forward_map(param, model) for param in eachcol(get_ϕ_final(prior, ekp))]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            throw("EKI terminated prematurely: $(terminate)! Shouldn't happen...")
        end
    end
    @info "EKP iterations: $n_iters"
    @info "Loss over iterations: $(get_error(ekp))"

    ekp_samples = Dict()
    for α in αs
        closest_iter = argmin(0:n_iters) do i
            abs(α - (i == 0 ? 0.0 : get_algorithm_time(ekp)[i]))
        end + 1
        ekp_samples[α] = (get_u(ekp, closest_iter), closest_iter == n_iters + 1 ? hcat([forward_map(u, model) for u in eachcol(get_u(ekp, closest_iter))]...) : get_g(ekp, closest_iter))
    end

    # [2] MCMC run
    mcmc_samples = Dict()
    for α in αs
        @info "Running MCMC for α = $α"

        prior_cov, prior_inv, obs_inv = cov(prior), inv(cov(prior)), inv(obs_noise_cov)
        mcmc_samples[α] = (zeros(input_dim, 0), zeros(output_dim, 0))
        do_mcmc(
            input_dim,
            x -> begin
                g = α == 0 ? 0y : forward_map(x, model)
                -2 \ x' * prior_inv * x - 2 \ (y - g)' * obs_inv * (y - g) * α
            end,
            step1_mcmc_num_chains,
            step1_mcmc_samples_per_chain,
            step1_mcmc_sampler,
            prior_cov,
            true_parameter;
            subsample_rate = step1_mcmc_subsample_rate,
        ) do samp, _
            gsamp = hcat([forward_map(s, model) for s in eachcol(samp)]...) # TODO: This is wasteful, as they have actually already been computed
            mcmc_samples[α] = (hcat(mcmc_samples[α][1], samp), hcat(mcmc_samples[α][2], gsamp))
        end
    end
    @info "MCMC finished"

    # [3] Save everything to a file
    #! format: off
    save(
        "datafiles/ekp_$(problem)_$(trial).jld2",
        "ekp", ekp,
        "ekp_samples", ekp_samples,
        "mcmc_samples", mcmc_samples,
        "prior", prior,
        "y", y,
        "obs_noise_cov", obs_noise_cov,
        "model", model,
        "true_parameter", true_parameter,
    )
    #! format: on
end
