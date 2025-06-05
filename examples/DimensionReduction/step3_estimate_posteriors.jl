using AdvancedMH
using Distributions
using ForwardDiff
using JLD2
using MCMCChains
using Plots
using Random
using Statistics

include("./settings.jl")
rng = Random.MersenneTwister(rng_seed)

if !isfile("datafiles/ekp_$(problem)_1.jld2")
    include("step1_generate_inverse_problem_data.jl")
end
if !isfile("datafiles/diagnostic_matrices_$(problem)_1.jld2")
    include("step2_build_and_compare_diagnostic_matrices.jl")
end

for (in_diag, in_r, out_diag, out_r) in step3_diagnostics_to_use
    @info "Diagnostic matrices = ($in_diag [1-$in_r], $out_diag [1-$out_r])"
    average_error = 0

    for trial in 1:num_trials
        # Load the EKP iterations
        loaded = load("datafiles/ekp_$(problem)_$(trial).jld2")
        ekp = loaded["ekp"]
        prior = loaded["prior"]
        obs_noise_cov = loaded["obs_noise_cov"]
        y = loaded["y"]
        model = loaded["model"]
        true_parameter = loaded["true_parameter"]

        prior_cov = cov(prior)
        prior_inv = inv(prior_cov)
        prior_invrt = sqrt(inv(prior_cov))
        prior_rt = sqrt(prior_cov)
        obs_rt = sqrt(obs_noise_cov)
        obs_invrt = sqrt(inv(obs_noise_cov))
        obs_inv = inv(obs_noise_cov)

        # Load diagnostic container
        diagnostic_mats = load("datafiles/diagnostic_matrices_$(problem)_$(trial).jld2")

        Hu = diagnostic_mats[in_diag]
        Hg = diagnostic_mats[out_diag]
        svdu = svd(Hu)
        svdg = svd(Hg)
        U_r = svdu.V[:, 1:in_r]
        V_r = svdg.V[:, 1:out_r]

        M = prior_rt * U_r * U_r' * prior_invrt
        N = obs_rt * V_r * V_r' * obs_invrt

        obs_noise_cov_r = V_r' * V_r # Vr' * invrt(noise) * noise * invrt(noise) * Vr
        obs_noise_cov_r_inv = inv(obs_noise_cov_r)
        prior_cov_r = U_r' * U_r
        prior_cov_r_inv = inv(prior_cov_r)
        y_r = V_r' * obs_invrt * y

        if step3_posterior_sampler == :mcmc
            logpostfull = x -> begin
                g = forward_map(x, model)
                -2\x'*prior_inv*x - 2\(y - g)'*obs_inv*(y - g)
            end
            densitymodelfull = DensityModel(logpostfull)
            mean_full = zeros(input_dim)
            num_iters = 1
            for _ in 1:num_iters
                sampler = if step3_mcmc_sampler == :mala
                    MALA(x -> MvNormal(.0001 * prior_cov * x, .0001 * 2 * prior_cov))
                elseif step3_mcmc_sampler == :rw
                    RWMH(MvNormal(zeros(input_dim), .01prior_cov))
                else
                    throw("Unknown step3_mcmc_sampler=$step3_mcmc_sampler")
                end
                chainfull = sample(densitymodelfull, sampler, MCMCThreads(), step3_mcmc_samples_per_chain, 8; chain_type=Chains, initial_params = [zeros(input_dim) for _ in 1:8])
                sampfull = vcat([vec(MCMCChains.get(chainfull, Symbol("param_$i"))[1])' for i in 1:input_dim]...)
                mean_full += mean(sampfull[:,end÷2:end]; dims = 2) / num_iters
            end
            mean_full_red = U_r' * (prior_invrt * mean_full)

            if step3_run_reduced_in_full_space
                logpostred = xfull -> begin
                    xredfull = M * xfull
                    gredfull = N * forward_map(xredfull, model)
            
                    -2\xfull'*prior_inv*xfull - 2\(y - gredfull)'*obs_inv*(y - gredfull)
                end
                densitymodelred = DensityModel(logpostred)

                mean_red_full = zeros(input_dim)
                num_iters = 1
                for _ in 1:num_iters
                    sampler = if step3_mcmc_sampler == :mala
                        MALA(x -> MvNormal(.0001 * prior_cov * x, .0001 * 2 * prior_cov))
                    elseif step3_mcmc_sampler == :rw
                        RWMH(MvNormal(zeros(input_dim), .01prior_cov))
                    else
                        throw("Unknown step3_mcmc_sampler=$step3_mcmc_sampler")
                    end
                    chainred = sample(densitymodelred, sampler, MCMCThreads(), step3_mcmc_samples_per_chain, 8; chain_type=Chains, initial_params = [zeros(input_dim) for _ in 1:8])
                    sampred = vcat([vec(MCMCChains.get(chainred, Symbol("param_$i"))[1])' for i in 1:input_dim]...)
                    mean_red_full += mean(sampred[:,end÷2:end]; dims = 2) / num_iters
                end
                mean_red = U_r' * (prior_invrt * mean_red_full)
            else
                logpostred = xred -> begin
                    xredfull = prior_rt * U_r * xred
                    gred = V_r' * obs_invrt * forward_map(xredfull, model)

                    -2\xred'*prior_cov_r_inv*xred - 2\(y_r - gred)'*obs_noise_cov_r_inv*(y_r - gred)
                end
                densitymodelred = DensityModel(logpostred)

                mean_red = zeros(in_r)
                num_iters = 1
                for _ in 1:num_iters
                    sampler, num_samples = if step3_mcmc_sampler == :mala
                        MALA(x -> MvNormal(.0001 * prior_cov_r * x, .0001 * 2 * prior_cov_r)), 50 # MALA is very slow, likely due to ForwardDiff
                    elseif step3_mcmc_sampler == :rw
                        RWMH(MvNormal(zeros(in_r), .01prior_cov_r)), 5_000
                    else
                        throw("Unknown step3_mcmc_sampler=$step3_mcmc_sampler")
                    end
                    chainred = sample(densitymodelred, sampler, MCMCThreads(), num_samples, 8; chain_type=Chains, initial_params = [zeros(in_r) for _ in 1:8])
                    sampred = vcat([vec(MCMCChains.get(chainred, Symbol("param_$i"))[1])' for i in 1:in_r]...)
                    mean_red += mean(sampred[:,end÷2:end]; dims = 2) / num_iters
                end
                mean_red_full = prior_rt * U_r * mean_red
            end
        elseif step3_posterior_sampler == :eks
            n_ensemble = step3_eks_ensemble_size
            n_iters_max = step3_eks_max_iters

            initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
            ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Sampler(prior); rng = rng, scheduler = EKSStableScheduler(2.0, 0.01))
            for i in 1:n_iters_max
                G_ens = hcat([forward_map(params, model) for params in eachcol(get_ϕ_final(prior, ekp))]...)
                isnothing(update_ensemble!(ekp, G_ens)) || break
            end
            ekp_u, ekp_g = reduce(hcat, get_u(ekp)), reduce(hcat, get_g(ekp))
            mean_full = get_u_mean_final(ekp)
            mean_full_red = U_r' * prior_invrt * mean_full

            if step3_run_reduced_in_full_space
                initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
                ekp_r = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Sampler(prior); rng, scheduler = EKSStableScheduler(2.0, 0.01))
        
                for i in 1:n_iters_max
                    G_ens = hcat([N*forward_map(M*params, model) for params in eachcol(get_ϕ_final(prior, ekp_r))]...)
                    isnothing(update_ensemble!(ekp_r, G_ens)) || break
                end
                ekp_r_u, ekp_r_g = reduce(hcat, get_u(ekp_r)), reduce(hcat, get_g(ekp_r))
                mean_red_full = get_u_mean_final(ekp_r)
                mean_red = U_r' * prior_invrt * mean_red_full
            else
                initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
                initial_r = U_r' * prior_invrt * initial_ensemble
                prior_r = ParameterDistribution(
                    Samples(U_r' * prior_invrt * sample(rng, prior, 1000)),
                    repeat([no_constraint()], in_r),
                    "prior_r",
                )

                ekp_r = EnsembleKalmanProcess(initial_r, y_r, obs_noise_cov_r, Sampler(mean(prior_r)[:], cov(prior_r)); rng)
        
                for i in 1:n_iters_max
                    # evaluate true G
                    G_ens_full = reduce(hcat, [forward_map(prior_rt * U_r * params, model) for params in eachcol(get_ϕ_final(prior_r, ekp_r))])
                    # project data back
                    G_ens = V_r' * obs_invrt * G_ens_full
        
                    isnothing(update_ensemble!(ekp_r, G_ens)) || break
                end
                ekp_r_u, ekp_r_g = reduce(hcat, get_u(ekp_r)), reduce(hcat, get_g(ekp_r))
                mean_red = get_u_mean_final(ekp_r)
                mean_red_full = prior_rt * U_r * mean_red
                # TODO: Check if we're OK with this way of projecting
            end
        else
            throw("Unknown step3_posterior_sampler=$step3_posterior_sampler")
        end

        @info """
        True: $(true_parameter[1:5])
        Mean (in full space):      $(mean_full[1:5])
        Red. mean (in full space): $(mean_red_full[1:5])
    
        Relative error in full space:    $(norm(mean_full - mean_red_full) / norm(mean_full))
        Relative error in reduced space: $(norm(mean_full_red - mean_red) / norm(mean_full_red))
        """
    end
end
