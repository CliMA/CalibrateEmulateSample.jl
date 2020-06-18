using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test

using CalibrateEmulateSample.GPEmulator
using CalibrateEmulateSample.MCMC

@testset "MCMC" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    # We need a GPObj to run MCMC, so let's reconstruct the one that's tested in 
    # runtets.jl in test/GPEmulator/
    # Training data
    n = 10                            # number of training points
    x = 2.0 * π * rand(n)             # predictors/features
    y = sin.(x) + 0.05 * randn(n)     # predictands/targets

    gppackage = GPJL()
    pred_type = YType()

    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    kern = SE(0.0, 0.0) 
    # log standard deviation of observational noise 
    logObsNoise = -1.0 
    white = Noise(logObsNoise)
    GPkernel = kern + white

    # Fit Gaussian Process Regression model
    gpobj = GPObj(x, y, gppackage; GPkernel=GPkernel, normalized=false, 
                  prediction_type=pred_type)

    ### Define prior
    umin = -1.0
    umax = 6.0
    prior = [Uniform(umin, umax)]
    obs_sample = [1.0]
    cov = convert(Array, Diagonal([1.0]))

    # MCMC parameters    
    mcmc_alg = "rwm" # random walk Metropolis
    param_init = [3.0] # set initial parameter 

    # First let's run a short chain to determine a good step size
    burnin = 0
    step = 0.5 # first guess
    max_iter = 5000
    mcmc_test = MCMCObj(obs_sample, cov, prior, step, param_init, max_iter, 
                        mcmc_alg, burnin)
    new_step = find_mcmc_step!(mcmc_test, gpobj)

    # reset parameters 
    burnin = 1000
    max_iter = 100_000

    # Now begin the actual MCMC
    mcmc = MCMCObj(obs_sample, cov, prior, step, param_init, max_iter, 
                   mcmc_alg, burnin)
    sample_posterior!(mcmc, gpobj, max_iter)
    posterior = get_posterior(mcmc)      
    post_mean = mean(posterior, dims=1)[1]

    @test mcmc.obs_sample == obs_sample
    @test mcmc.obs_cov == cov
    @test mcmc.obs_covinv == inv(cov)
    @test mcmc.burnin == burnin
    @test mcmc.algtype == mcmc_alg
    @test mcmc.iter[1] == max_iter+ 1
    @test size(posterior) == (max_iter - burnin + 1, length(param_init))
    @test_throws Exception MCMCObj(obs_sample, cov, prior, step, param_init, 
                                   max_iter, "gibbs", burnin)
    @test post_mean ≈ π/2 atol=3e-1

end
