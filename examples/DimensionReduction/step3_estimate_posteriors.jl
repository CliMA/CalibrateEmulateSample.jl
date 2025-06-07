using AdvancedMH
using Distributions
using ForwardDiff
using JLD2
using LinearAlgebra
using MCMCChains
using Plots
using Random
using Statistics

include("./settings.jl")
include("./util.jl")
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
        obs_invrt = sqrt(inv(obs_noise_cov))
        obs_inv = inv(obs_noise_cov)

        # Load diagnostic container
        diagnostic_mats = load("datafiles/diagnostic_matrices_$(problem)_$(trial).jld2")

        Hu = diagnostic_mats[in_diag]
        Hg = diagnostic_mats[out_diag]
        svdu = svd(Hu; alg=LinearAlgebra.QRIteration())
        svdg = svd(Hg; alg=LinearAlgebra.QRIteration())
        U_r = svdu.V[:, 1:in_r]
        V_r = svdg.V[:, 1:out_r]

        # Projection matrices
        P = U_r' * prior_invrt
        Pinv = prior_rt * U_r
        Q = V_r' * obs_invrt

        obs_noise_cov_r = V_r' * V_r # Vr' * invrt(noise) * noise * invrt(noise) * Vr
        obs_noise_cov_r_inv = inv(obs_noise_cov_r)
        prior_cov_r = U_r' * U_r
        prior_cov_r_inv = inv(prior_cov_r)
        y_r = V_r' * obs_invrt * y

        # TODO: Fix assert below for the actual type of `prior`
        # @assert prior isa MvNormal && mean(prior) == zeros(input_dim)

        # Let prior = N(0, C) and let x ~ prior
        # Then the distribution of x | P*x=x_r is N(Mmean * x_r, Mcov)
        C = prior_cov
        Mmean = C*P'*inv(P*C*P')
        @assert Pinv ≈ Mmean
        Mcov = C - Mmean*P*C + 1e-13 * I
        Mcov = (Mcov + Mcov') / 2 # Otherwise, it's not numerically Hermitian
        covsamps = rand(MvNormal(zeros(input_dim), Mcov), 8)

        if step3_posterior_sampler == :mcmc
            mean_full = zeros(input_dim)
            do_mcmc(input_dim, x -> begin
                g = forward_map(x, model)
                -2\x'*prior_inv*x - 2\(y - g)'*obs_inv*(y - g)
            end, step3_mcmc_num_chains, step3_mcmc_samples_per_chain, step3_mcmc_sampler, prior_cov) do samp, num_batches
                mean_full += mean(samp; dims = 2) / num_batches
            end
            mean_full_red = P * mean_full

            if step3_run_reduced_in_full_space
                mean_red_full = zeros(input_dim)
                do_mcmc(input_dim, xfull -> begin
                    xred = P*xfull
                    samp = covsamps .+ Mmean * xred
                    gsamp = map(x -> forward_map(x, model), eachcol(samp))

                    return -2\xfull'*prior_inv*xfull + mean(
                        -2\(Q*(y - g))'*inv(Q*obs_noise_cov*Q')*(Q*(y - g))
                        for (x, g) in zip(eachcol(samp), gsamp)
                    )
                end, step3_mcmc_num_chains, step3_mcmc_samples_per_chain, step3_mcmc_sampler, prior_cov) do samp, num_batches
                    mean_red_full += mean(samp; dims = 2) / num_batches
                end
                mean_red = P * mean_red_full
            else
                mean_red = zeros(in_r)
                do_mcmc(in_r, xred -> begin
                    samp = covsamps .+ Mmean * xred
                    gsamp = map(x -> forward_map(x, model), eachcol(samp))

                    return -2\xred'*prior_cov_r_inv*xred + mean(
                        -2\(Q*(y - g))'*inv(Q*obs_noise_cov*Q')*(Q*(y - g))
                        for (x, g) in zip(eachcol(samp), gsamp)
                    )
                end, step3_mcmc_num_chains, step3_mcmc_samples_per_chain, step3_mcmc_sampler, prior_cov_r) do samp, num_batches
                    mean_red += mean(samp; dims = 2) / num_batches
                end
                mean_red_full = Pinv*mean_red # This only works since it's the mean (linear) — if not, we'd have to use the covsamps here
            end
        elseif step3_posterior_sampler == :eks
            throw("""
                EKS sampling from the reduced posterior is currently not supported:
                 The reduced posterior density is not straightforwardly defined in terms of a forward model.
                 We need to look into this.
            """)
            # n_ensemble = step3_eks_ensemble_size
            # n_iters_max = step3_eks_max_iters

            # initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
            # ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Sampler(prior); rng = rng, scheduler = EKSStableScheduler(2.0, 0.01))
            # for i in 1:n_iters_max
            #     G_ens = hcat([forward_map(params, model) for params in eachcol(get_ϕ_final(prior, ekp))]...)
            #     isnothing(update_ensemble!(ekp, G_ens)) || break
            # end
            # ekp_u, ekp_g = reduce(hcat, get_u(ekp)), reduce(hcat, get_g(ekp))
            # mean_full = get_u_mean_final(ekp)
            # mean_full_red = U_r' * prior_invrt * mean_full

            # if step3_run_reduced_in_full_space
            #     initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
            #     ekp_r = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Sampler(prior); rng, scheduler = EKSStableScheduler(2.0, 0.01))
        
            #     for i in 1:n_iters_max
            #         G_ens = hcat([N*forward_map(M*params, model) for params in eachcol(get_ϕ_final(prior, ekp_r))]...)
            #         isnothing(update_ensemble!(ekp_r, G_ens)) || break
            #     end
            #     ekp_r_u, ekp_r_g = reduce(hcat, get_u(ekp_r)), reduce(hcat, get_g(ekp_r))
            #     mean_red_full = get_u_mean_final(ekp_r)
            #     mean_red = U_r' * prior_invrt * mean_red_full
            # else
            #     initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
            #     initial_r = U_r' * prior_invrt * initial_ensemble
            #     prior_r = ParameterDistribution(
            #         Samples(U_r' * prior_invrt * sample(rng, prior, 1000)),
            #         repeat([no_constraint()], in_r),
            #         "prior_r",
            #     )

            #     ekp_r = EnsembleKalmanProcess(initial_r, y_r, obs_noise_cov_r, Sampler(mean(prior_r)[:], cov(prior_r)); rng)
        
            #     for i in 1:n_iters_max
            #         # evaluate true G
            #         G_ens_full = reduce(hcat, [forward_map(prior_rt * U_r * params, model) for params in eachcol(get_ϕ_final(prior_r, ekp_r))])
            #         # project data back
            #         G_ens = V_r' * obs_invrt * G_ens_full
        
            #         isnothing(update_ensemble!(ekp_r, G_ens)) || break
            #     end
            #     ekp_r_u, ekp_r_g = reduce(hcat, get_u(ekp_r)), reduce(hcat, get_g(ekp_r))
            #     mean_red = get_u_mean_final(ekp_r)
            #     mean_red_full = prior_rt * U_r * mean_red
            # end
        else
            throw("Unknown step3_posterior_sampler=$step3_posterior_sampler")
        end

        @info """
        True: $(true_parameter[1:5])
        Mean (in full space):      $(mean_full[1:5])
        Red. mean (in full space): $(mean_red_full[1:5])

        Mean (in red. space):      $(mean_full_red[1:16])
        Red. mean (in red. space): $(mean_red[1:16])
    
        Relative error on mean in full space:    $(norm(mean_full - mean_red_full) / norm(mean_full))
        Relative error on mean in reduced space: $(norm(mean_full_red - mean_red) / norm(mean_full_red))
        """
        # [A] The relative error seems larger in the reduced space
        #     The reason is likely the whitening that happens. Small absolute errors in the full space
        #     can be amplified in the reduced space due to the different scales in the prior. I think
        #     the full space is probably the one we should be concerned about.
    end
end
