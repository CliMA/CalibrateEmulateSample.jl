using AdvancedMH
using Distributions
using EnsembleKalmanProcesses
using ForwardDiff
using JLD2
using LinearAlgebra
using MCMCChains
using Random

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

    # [1] EKP run
    n_ensemble = step1_eki_ensemble_size
    n_iters_max = step1_eki_max_iters

    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, TransformInversion(); rng, scheduler = EKSStableScheduler(2.0, 0.01))

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
    logpost = x -> begin
        g = forward_map(x, model)
        (-2\x'*prior_inv*x - 2\(y - g)'*obs_inv*(y - g)) / step1_mcmc_temperature
    end
    density_model = DensityModel(logpost)
    num_iters = 1
    mcmc_samples = zeros(input_dim, 0)
    for _ in 1:num_iters
        sampler = if step1_mcmc_sampler == :mala
            MALA(x -> MvNormal(.0001 * prior_cov * x, .0001 * 2 * prior_cov))
        elseif step1_mcmc_sampler == :rw
            RWMH(MvNormal(zeros(input_dim), .01prior_cov))
        else
            throw("Unknown step1_mcmc_sampler=$step1_mcmc_sampler")
        end
        chain = sample(density_model, sampler, MCMCThreads(), step1_mcmc_samples_per_chain, 8; chain_type=Chains, initial_params=[zeros(input_dim) for _ in 1:8])
        samp = vcat([vec(MCMCChains.get(chain, Symbol("param_$i"))[1]'[:, end÷2:step1_mcmc_subsample_rate:end])' for i in 1:input_dim]...)
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
