using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test

using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.GaussianProcessEmulator
using CalibrateEmulateSample.DataContainers

@testset "MarkovChainMonteCarlo" begin

    @testset "MCMC_base" begin
        # Seed for pseudo-random number generator
        rng_seed = 41
        Random.seed!(rng_seed)

        # We need a GaussianProcess to run MarkovChainMonteCarlo, so let's reconstruct the 
        # one that's tested in test/GaussianProcesses/runtests.jl
        n = 20                                       # number of training points
        x = reshape(2.0 * π * rand(n), 1, n)         # predictors/features: 1 × n
        σ2_y = reshape([0.05],1,1)
        y = sin.(x) + rand(Normal(0, σ2_y[1]), (1, n)) # predictands/targets: 1 × n

        iopairs = PairedDataContainer(x,y,data_are_columns=true)

        gppackage = GPJL()
        pred_type = YType()

        obs_sample = [1.0]
        # observational noise 
        obs_noise = 0.1

        # Construct kernel:
        # Squared exponential kernel (note that hyperparameters are on log scale)
        # with observational noise
        GPkernel = SE(log(1.0), log(1.0))

        gp = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, obs_noise_cov=σ2_y, 
                    normalized=false, noise_learn=true, 
                    prediction_type=pred_type) 

        ### Define prior
        umin = -1.0
        umax = 6.0
        #prior = [Priors.Prior(Uniform(umin, umax), "u")]  # prior on u
        prior_dist = Parameterized(Normal(0,1))
        prior_constraint = bounded(-1.0,6.0)
        prior_name = "u"
        prior = ParameterDistribution(prior_dist, prior_constraint, prior_name)

        # First let's run a short chain to determine a good step size
        mcmc_1 = MCMCWrapper(
            GaussianProcessRWSampling(), # random walk Metropolis
            obs_sample, σ2_y, gp, prior;
            step = 0.5,         # first guess
            param_init = [3.0], # set initial parameter 
            norm_factor = nothing,
            svd = true,
            truncate_svd = 1.0
        )
        new_step = find_mcmc_step!(mcmc_1)

        # Now begin the actual MCMC
        burnin = 1000
        max_iter = 100_000
        chain = sample(mcmc_1, max_iter; discard_initial = burnin)
        posterior_distribution = get_posterior(mcmc_1, chain)      
        #post_mean = mean(posterior, dims=1)[1]
        posterior_mean = mean(posterior_distribution)
        # We had svdflag=true, so the MCMC stores a transformed sample, 
        # 1.0/sqrt(0.05) * obs_sample ≈ 4.472
        @test mcmc.obs_sample ≈ [4.472] atol=1e-2
        @test mcmc.obs_noise_cov == σ2_y
        @test mcmc.burnin == burnin
        @test mcmc.iter[1] == max_iter + 1
        @test get_n_samples(posterior_distribution)[prior_name] == max_iter - burnin + 1
        @test ndims(posterior_distribution) == length(param_init)
        @test_throws Exception MCMC(obs_sample, σ2_y, prior, step, param_init, 
                                    max_iter, "gibbs", burnin)
        @test isapprox(posterior_mean[1] - π/2, 0.0; atol=4e-1)
    end



    @testset "MCMC_standardization" begin
        # Standardization and truncation
        norm_factor = 10.0
        norm_factor = fill(norm_factor, size(y[:,1])) # must be size of output dim
        gp = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, obs_noise_cov=σ2_y, 
                    normalized=false, noise_learn=true, standardize=true, truncate_svd=0.9, 
                    prediction_type=pred_type, norm_factor=norm_factor) 
        mcmc_test = MCMC(obs_sample, σ2_y, prior, step, param_init, max_iter, 
                            mcmc_alg, burnin;
                svdflag=true, standardize=true, norm_factor=norm_factor, truncate_svd=0.9)

        # Now begin the actual MCMC
        mcmc = MCMC(obs_sample, σ2_y, prior, step, param_init, max_iter, 
                    mcmc_alg, burnin; svdflag=true, standardize=false,
                    truncate_svd=1.0, norm_factor=norm_factor)
        sample_posterior!(mcmc, gp, max_iter)
        posterior_distribution = get_posterior(mcmc)      
        #post_mean = mean(posterior, dims=1)[1]
        posterior_mean2 = mean(posterior_distribution)
        @test posterior_mean2 ≈ posterior_mean atol=0.1

    end
end
