using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test

using CalibrateEmulateSample.MCMC
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.GPEmulator

@testset "MCMC" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    # We need a GPObj to run MCMC, so let's reconstruct the one that's tested
    # in test/GPEmulator/runtests.jl
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), n, 1)         # predictors/features: n x 1
    σ2_y = reshape([0.05], 1, 1)
    y = reshape(sin.(x) + rand(Normal(0, σ2_y[1]), n), n, 1) # predictands/targets: n x 1

    gppackage = GPJL()
    pred_type = YType()

    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))

    gpobj = GPObj(x, y, gppackage; GPkernel=GPkernel, obs_noise_cov=σ2_y, 
                  normalized=false, noise_learn=true, 
                  prediction_type=pred_type) 

    ### Define prior
    umin = -1.0
    umax = 6.0
    prior = [Priors.Prior(Uniform(umin, umax), "u")]  # prior on u
    obs_sample = [1.0]

    # MCMC parameters    
    mcmc_alg = "rwm" # random walk Metropolis
    param_init = [3.0] # set initial parameter 

    # First let's run a short chain to determine a good step size
    burnin = 0
    step = 0.5 # first guess
    max_iter = 5000
    mcmc_test = MCMCObj(obs_sample, σ2_y, prior, step, param_init, max_iter, 
                        mcmc_alg, burnin)
    new_step = find_mcmc_step!(mcmc_test, gpobj)

    # reset parameters 
    burnin = 1000
    max_iter = 100_000

    # Now begin the actual MCMC
    mcmc = MCMCObj(obs_sample, σ2_y, prior, step, param_init, max_iter, 
                   mcmc_alg, burnin, svdflag=true)
    sample_posterior!(mcmc, gpobj, max_iter)
    posterior = get_posterior(mcmc)      
    post_mean = mean(posterior, dims=1)[1]
    
    # We had svdflag=true, so the MCMCObj stores a transformed sample, 
    # 1.0/sqrt(0.05) * obs_sample ≈ 4.472
    @test mcmc.obs_sample ≈ [4.472] atol=1e-2
    @test mcmc.obs_noise_cov == σ2_y
    @test mcmc.obs_noise_covinv == inv(σ2_y)
    @test mcmc.burnin == burnin
    @test mcmc.algtype == mcmc_alg
    @test mcmc.iter[1] == max_iter + 1
    @test size(posterior) == (max_iter - burnin + 1, length(param_init))
    @test_throws Exception MCMCObj(obs_sample, σ2_y, prior, step, param_init, 
                                   max_iter, "gibbs", burnin)
    @test post_mean ≈ π/2 atol=3e-1

end
