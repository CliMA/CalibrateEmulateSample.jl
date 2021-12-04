using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test

using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers

function test_gp_data(; rng_seed = 41, n = 20, var_y = 0.05, rest...)
    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)
    # We need a GaussianProcess to run MarkovChainMonteCarlo, so let's reconstruct the one 
    # that's tested in test/GaussianProcesses/runtests.jl Case 1
    n = 40                                              # number of training points
    x = 2π * rand(rng, Float64, (1, n))                 # predictors/features: 1 × n
    σ2_y = reshape([var_y], 1, 1)
    y = sin.(x) + rand(rng, Normal(0, σ2_y[1]), (1, n)) # predictands/targets: 1 × n

    return y, σ2_y, PairedDataContainer(x, y, data_are_columns = true), rng
end

function test_gp_prior()
    ### Define prior
    umin = -1.0
    umax = 6.0
    #prior = [Priors.Prior(Uniform(umin, umax), "u")]  # prior on u
    prior_dist = Parameterized(Normal(0, 1))
    prior_constraint = bounded(umin, umax)
    prior_name = "u"
    return ParameterDistribution(prior_dist, prior_constraint, prior_name)
end

function test_gp_1(y, σ2_y, iopairs::PairedDataContainer; norm_factor = nothing)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))
    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    em = Emulator(
        gp,
        iopairs;
        obs_noise_cov = σ2_y,
        normalize_inputs = false,
        standardize_outputs = false,
        standardize_outputs_factors = norm_factor,
        truncate_svd = 1.0,
    )
    Emulators.optimize_hyperparameters!(em)
    return em
end

function test_gp_2(y, σ2_y, iopairs::PairedDataContainer; norm_factor = nothing)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))
    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    em = Emulator(
        gp,
        iopairs;
        obs_noise_cov = σ2_y,
        normalize_inputs = false,
        standardize_outputs = true,
        standardize_outputs_factors = norm_factor,
        truncate_svd = 0.9,
    )
    Emulators.optimize_hyperparameters!(em)
    return em
end

function mcmc_gp_test_step(
    prior::ParameterDistribution,
    σ2_y,
    em::Emulator;
    mcmc_alg = "rwm",
    obs_sample = 1.0,
    param_init = 3.0,
    step = 0.5,
    norm_factor = nothing,
    rng = Random.GLOBAL_RNG,
)
    obs_sample = reshape(collect(obs_sample), 1) # scalar or Vector -> Vector
    param_init = reshape(collect(param_init), 1) # scalar or Vector -> Vector
    # First let's run a short chain to determine a good step size
    burnin = 0
    step = 0.5 # first guess
    max_iter = 5000
    mcmc_test = MCMC(
        obs_sample,
        σ2_y,
        prior,
        step,
        param_init,
        max_iter,
        mcmc_alg,
        burnin;
        svdflag = true,
        standardize = false,
        truncate_svd = 1.0,
        norm_factor = nothing,
        rng = rng,
    )
    new_step = find_mcmc_step!(mcmc_test, em)
    return new_step
end

function mcmc_gp_test_template(
    prior::ParameterDistribution,
    σ2_y,
    em::Emulator;
    mcmc_alg = "rwm",
    obs_sample = 1.0,
    param_init = 3.0,
    step = 0.5,
    norm_factor = nothing,
    rng = Random.GLOBAL_RNG,
)
    obs_sample = reshape(collect(obs_sample), 1) # scalar or Vector -> Vector
    param_init = reshape(collect(param_init), 1) # scalar or Vector -> Vector
    # reset parameters 
    burnin = 1000
    max_iter = 100_000
    # Now begin the actual MCMC
    mcmc = MCMC(
        obs_sample,
        σ2_y,
        prior,
        step,
        param_init,
        max_iter,
        mcmc_alg,
        burnin;
        svdflag = true,
        standardize = false,
        truncate_svd = 1.0,
        norm_factor = norm_factor,
        rng = rng,
    )
    sample_posterior!(mcmc, em, max_iter)
    posterior_distribution = get_posterior(mcmc)
    #post_mean = mean(posterior, dims=1)[1]
    posterior_mean = mean(posterior_distribution)
    # We had svdflag=true, so the MCMC stores a transformed sample, 
    # 1.0/sqrt(0.05) * obs_sample ≈ 4.472
    @test mcmc.obs_sample ≈ (obs_sample ./ sqrt(σ2_y[1, 1])) atol = 1e-2
    @test mcmc.obs_noise_cov == σ2_y
    @test mcmc.burnin == burnin
    @test mcmc.algtype == mcmc_alg
    @test mcmc.iter[1] == max_iter + 1
    @test get_n_samples(posterior_distribution)[prior.names[1]] == max_iter - burnin + 1
    @test ndims(posterior_distribution) == length(param_init)
    @test_throws Exception MCMC(obs_sample, σ2_y, prior, step, param_init, max_iter, "gibbs", burnin)
    return posterior_mean[1]
end

function mcmc_gp_standardization_test_template(
    prior::ParameterDistribution,
    σ2_y,
    em::Emulator;
    mcmc_alg = "rwm",
    obs_sample = 1.0,
    param_init = 3.0,
    step = 0.5,
    norm_factor = nothing,
    rng = Random.GLOBAL_RNG,
)
    obs_sample = reshape(collect(obs_sample), 1) # scalar or Vector -> Vector
    param_init = reshape(collect(param_init), 1) # scalar or Vector -> Vector
    # reset parameters 
    burnin = 1000
    max_iter = 100_000
    # Standardization and truncation
    mcmc_test = MCMC(
        obs_sample,
        σ2_y,
        prior,
        step,
        param_init,
        max_iter,
        mcmc_alg,
        burnin;
        svdflag = true,
        standardize = true,
        norm_factor = norm_factor,
        truncate_svd = 0.9,
        rng = rng,
    )
    # Now begin the actual MCMC
    mcmc = MCMC(
        obs_sample,
        σ2_y,
        prior,
        step,
        param_init,
        max_iter,
        mcmc_alg,
        burnin;
        svdflag = true,
        standardize = false,
        truncate_svd = 1.0,
        norm_factor = norm_factor,
        rng = rng,
    )
    sample_posterior!(mcmc, em, max_iter)
    posterior_mean = mean(get_posterior(mcmc))
    return posterior_mean[1]
end

@testset "MarkovChainMonteCarlo" begin
    @testset "GP & rwm" begin
        prior = test_gp_prior()
        y, σ2_y, iopairs, rng = test_gp_data(; rng_seed = 456, n = 20, var_y = 0.05)
        em = test_gp_1(y, σ2_y, iopairs)
        mcmc_params = Dict(
            :mcmc_alg => "rwm",
            :obs_sample => 1.0,
            :param_init => 3.0,
            :step => 0.5,
            :norm_factor => nothing,
            :rng => rng,
        )

        new_step = mcmc_gp_test_step(prior, σ2_y, em; mcmc_params...)
        @test isapprox(new_step, 1.0; atol = 0.1)

        posterior_mean_1 = mcmc_gp_test_template(prior, σ2_y, em; mcmc_params...)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        norm_factor = 10.0
        norm_factor = fill(norm_factor, size(y[:, 1])) # must be size of output dim
        em = test_gp_2(y, σ2_y, iopairs; norm_factor = norm_factor)
        posterior_mean_2 =
            mcmc_gp_standardization_test_template(prior, σ2_y, em; mcmc_params..., norm_factor = norm_factor)

        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
    end

    @testset "GP & pCN" begin
        prior = test_gp_prior()
        y, σ2_y, iopairs, rng = test_gp_data(; rng_seed = 456, n = 20, var_y = 0.05)
        em = test_gp_1(y, σ2_y, iopairs)
        mcmc_params = Dict(
            :mcmc_alg => "pCN",
            :obs_sample => 1.0,
            :param_init => 3.0,
            :step => 0.5,
            :norm_factor => nothing,
            :rng => rng,
        )

        new_step = mcmc_gp_test_step(prior, σ2_y, em; mcmc_params...)
        @test isapprox(new_step, 1.0; atol = 0.1)

        posterior_mean_1 = mcmc_gp_test_template(prior, σ2_y, em; mcmc_params...)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        norm_factor = 10.0
        norm_factor = fill(norm_factor, size(y[:, 1])) # must be size of output dim
        em = test_gp_2(y, σ2_y, iopairs; norm_factor = norm_factor)
        posterior_mean_2 =
            mcmc_gp_standardization_test_template(prior, σ2_y, em; mcmc_params..., norm_factor = norm_factor)

        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
    end
end
