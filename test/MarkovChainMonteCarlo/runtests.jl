using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test

using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributionStorage
using CalibrateEmulateSample.GaussianProcessEmulator
using CalibrateEmulateSample.DataStorage

@testset "MarkovChainMonteCarlo" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    # We need a GaussianProcess to run MarkovChainMonteCarlo, so let's reconstruct the one that's tested
    # in test/GaussianProcesses/runtests.jl
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), 1, n)         # predictors/features: 1 × n
    σ2_y = reshape([0.05],1,1)
    y = sin.(x) + rand(Normal(0, σ2_y[1]), (1, n)) # predictands/targets: 1 × n

    iopairs = PairedDataContainer(x,y,data_are_columns=true)

    gppackage = GPJL()
    pred_type = YType()

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
    
    obs_sample = [1.0]

    # MarkovChainMonteCarlo parameters    
    mcmc_alg = "rwm" # random walk Metropolis
    param_init = [3.0] # set initial parameter 

    # First let's run a short chain to determine a good step size
    burnin = 0
    step = 0.5 # first guess
    max_iter = 5000
    norm_factors=nothing
    mcmc_test = MCMC(obs_sample, σ2_y, prior, step, param_init, max_iter, 
                        mcmc_alg, burnin, norm_factors; svdflag=true, standardize=false,
			truncate_svd=1.0)
    new_step = find_mcmc_step!(mcmc_test, gp)

    # reset parameters 
    burnin = 1000
    max_iter = 100_000

    # Now begin the actual MCMC
    mcmc = MCMC(obs_sample, σ2_y, prior, step, param_init, max_iter, 
                   mcmc_alg, burnin, norm_factors; svdflag=true, standardize=false,
                   truncate_svd=1.0)
    sample_posterior!(mcmc, gp, max_iter)
    posterior_distribution = get_posterior(mcmc)      
    #post_mean = mean(posterior, dims=1)[1]
    posterior_mean = get_mean(posterior_distribution)
    # We had svdflag=true, so the MCMC stores a transformed sample, 
    # 1.0/sqrt(0.05) * obs_sample ≈ 4.472
    @test mcmc.obs_sample ≈ [4.472] atol=1e-2
    @test mcmc.obs_noise_cov == σ2_y
    @test mcmc.burnin == burnin
    @test mcmc.algtype == mcmc_alg
    @test mcmc.iter[1] == max_iter + 1
    @test get_n_samples(posterior_distribution)[prior_name] == max_iter - burnin + 1
    @test get_total_dimension(posterior_distribution) == length(param_init)
    @test_throws Exception MCMC(obs_sample, σ2_y, prior, step, param_init, 
                                   max_iter, "gibbs", burnin)
    @test isapprox(posterior_mean[1] - π/2, 0.0; atol=4e-1)

end
