using AdvancedMH
using Distributions
using ForwardDiff
using MCMCChains

function do_mcmc(callback, dim, logpost, num_chains, num_samples_per_chain, mcmc_sampler, prior_cov; subsample_rate=1)
    density_model = DensityModel(logpost)
    sampler = if mcmc_sampler == :mala
        MALA(x -> MvNormal(.0001 * prior_cov * x, .0001 * 2 * prior_cov))
    elseif mcmc_sampler == :rw
        RWMH(MvNormal(zeros(dim), .01prior_cov))
    else
        throw("Unknown mcmc_sampler=$mcmc_sampler")
    end

    num_batches = (num_chains + 7) ÷ 8
    for batch in 1:num_batches
        num_chains_in_batch = min(8, num_chains - (batch - 1)*8)
        chain = sample(density_model, sampler, MCMCThreads(), num_samples_per_chain, num_chains_in_batch; chain_type=Chains, initial_params=[zeros(dim) for _ in 1:num_chains_in_batch])
        samp = vcat([vec(MCMCChains.get(chain, Symbol("param_$i"))[1]'[:, end÷2:subsample_rate:end])' for i in 1:dim]...)
        callback(samp, num_batches)
    end
end

function do_eks(dim, G, y, obs_noise_cov, prior, rng, num_ensemble, num_iters_max)
    initial_ensemble = construct_initial_ensemble(rng, prior, num_ensemble)
    ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Sampler(prior); rng, scheduler=EKSStableScheduler(2.0, 0.01))

    for i in 1:num_iters_max
        g = hcat([G(params) for params in eachcol(get_ϕ_final(prior, ekp))]...)
        isnothing(update_ensemble!(ekp, g)) || break
    end

    return get_u_final(ekp), get_g_final(ekp)
end
