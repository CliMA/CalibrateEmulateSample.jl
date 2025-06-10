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
        @assert obs_noise_cov_r ≈ I
        prior_cov_r = U_r' * U_r
        prior_cov_r_inv = inv(prior_cov_r)
        y_r = Q * y
        prior_r = ParameterDistribution(
            Dict(
                "distribution" => Parameterized(MvNormal(zeros(in_r), prior_cov_r)),
                "constraint" => repeat([no_constraint()], in_r),
                "name" => "param_$(in_r)",
            ),
        )

        # TODO: Fix assert below for the actual type of `prior`
        # @assert prior isa MvNormal && mean(prior) == zeros(input_dim)

        # Let prior = N(0, C) and let x ~ prior
        # Then the distribution of x | P*x=x_r is N(Mmean * x_r, Mcov)
        C = prior_cov
        Mmean = C*P'*inv(P*C*P')
        @assert Pinv ≈ Mmean
        Mcov = C - Mmean*P*C + 1e-13 * I
        Mcov = (Mcov + Mcov') / 2 # Otherwise, it's not numerically Hermitian
        covsamps = rand(MvNormal(zeros(input_dim), Mcov), step3_num_marginalization_samples)

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

                    return -2\xfull'*prior_inv*xfull + if step3_marginalization == :loglikelihood
                        mean(
                            -2\(y_r - Q*g)'*(y_r - Q*g)
                            for (x, g) in zip(eachcol(samp), gsamp)
                        )
                    elseif step3_marginalization == :forward_model
                        g = mean(gsamp)
                        -2\(y_r - Q*g)'*(y_r - Q*g)
                    else
                        throw("Unknown step3_marginalization=$step3_marginalization")
                    end
                end, step3_mcmc_num_chains, step3_mcmc_samples_per_chain, step3_mcmc_sampler, prior_cov) do samp, num_batches
                    mean_red_full += mean(samp; dims = 2) / num_batches
                end
                mean_red = P * mean_red_full
            else
                mean_red = zeros(in_r)
                do_mcmc(in_r, xred -> begin
                    samp = covsamps .+ Mmean * xred
                    gsamp = map(x -> forward_map(x, model), eachcol(samp))

                    return -2\xred'*prior_cov_r_inv*xred + if step3_marginalization == :loglikelihood
                        mean(
                            -2\(y_r - Q*g)'*(y_r - Q*g)
                            for (x, g) in zip(eachcol(samp), gsamp)
                        )
                    elseif step3_marginalization == :forward_model
                        g = mean(gsamp)
                        -2\(y_r - Q*g)'*(y_r - Q*g)
                    else
                        throw("Unknown step3_marginalization=$step3_marginalization")
                    end
                end, step3_mcmc_num_chains, step3_mcmc_samples_per_chain, step3_mcmc_sampler, prior_cov_r) do samp, num_batches
                    mean_red += mean(samp; dims = 2) / num_batches
                end
                mean_red_full = Pinv*mean_red # This only works since it's the mean (linear) — if not, we'd have to use the covsamps here (same in a few other places)
            end
        elseif step3_posterior_sampler == :eks
            step3_marginalization == :forward_model || throw("EKS sampling from the reduced posterior is only supported when marginalizing over the forward model.")

            u, _ = do_eks(input_dim, x -> forward_map(x, model), y, obs_noise_cov, prior, rng, step3_eks_ensemble_size, step3_eks_max_iters)
            mean_full = mean(u; dims = 2)
            mean_full_red = P * mean_full

            if step3_run_reduced_in_full_space
                u, _ = do_eks(input_dim, xfull -> begin
                    xred = P*xfull
                    samp = covsamps .+ Mmean * xred
                    gsamp = map(x -> forward_map(x, model), eachcol(samp))
                    return Q*mean(gsamp)
                end, y_r, 1.0*I(out_r), prior, rng, step3_eks_ensemble_size, step3_eks_max_iters)
                mean_red_full = mean(u; dims = 2)
                mean_red = P * mean_red_full
            else
                u, _ = do_eks(in_r, xred -> begin
                    samp = covsamps .+ Mmean * xred
                    gsamp = map(x -> forward_map(x, model), eachcol(samp))
                    return Q*mean(gsamp)
                end, y_r, 1.0*I(out_r), prior_r, rng, step3_eks_ensemble_size, step3_eks_max_iters)
                mean_red = mean(u; dims = 2)
                mean_red_full = Pinv*mean_red
            end
        else
            throw("Unknown step3_posterior_sampler=$step3_posterior_sampler")
        end

        @info """
        True: $(true_parameter[1:5])
        Mean (in full space):      $(mean_full[1:5])
        Red. mean (in full space): $(mean_red_full[1:5])

        Mean (in red. space):      $(mean_full_red)
        Red. mean (in red. space): $(mean_red)
    
        Relative error on mean in full space:    $(norm(mean_full - mean_red_full) / norm(mean_full))
        Relative error on mean in reduced space: $(norm(mean_full_red - mean_red) / norm(mean_full_red))
        """
        # [A] The relative error seems larger in the reduced space
        #     The reason is likely the whitening that happens. Small absolute errors in the full space
        #     can be amplified in the reduced space due to the different scales in the prior. I think
        #     the full space is probably the one we should be concerned about.
    end
end
