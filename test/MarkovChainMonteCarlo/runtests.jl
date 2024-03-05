using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test

using CalibrateEmulateSample.MarkovChainMonteCarlo
const MCMC = MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers

function test_data(; rng_seed = 41, n = 20, var_y = 0.05, rest...)
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

function test_prior()
    ### Define prior
    umin = -1.0
    umax = 6.0
    #prior = [Priors.Prior(Uniform(umin, umax), "u")]  # prior on u
    prior_dist = Parameterized(Normal(0, 1))
    prior_constraint = bounded(umin, umax)
    prior_name = "u"
    return ParameterDistribution(prior_dist, prior_constraint, prior_name)
end

function test_prior_mv()
    ### Define prior
    return constrained_gaussian("u_mv", -1.0, 6.0, -Inf, Inf, repeats = 10)
end

function test_data_mv(; rng_seed = 41, n = 20, var_y = 0.05, input_dim = 10, rest...)
    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)
    n = 40                                              # number of training points
    x = 1 / input_dim * π * rand(rng, Float64, (input_dim, n))                 # predictors/features: 1 × n
    σ2_y = reshape([var_y], 1, 1)
    y = sin.(norm(x)) + rand(rng, Normal(0, σ2_y[1]), (1, n)) # predictands/targets: 1 × n

    return y, σ2_y, PairedDataContainer(x, y, data_are_columns = true), rng
end
function test_gp_mv(y, σ2_y, iopairs::PairedDataContainer; norm_factor = nothing)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    gp = GaussianProcess(gppackage; noise_learn = true, prediction_type = pred_type)
    em = Emulator(gp, iopairs; obs_noise_cov = σ2_y)
    Emulators.optimize_hyperparameters!(em)
    return em
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
        retained_svd_frac = 1.0,
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
        retained_svd_frac = 0.9,
    )
    Emulators.optimize_hyperparameters!(em)
    return em
end

function mcmc_test_template(
    prior::ParameterDistribution,
    σ2_y,
    em::Emulator;
    mcmc_alg = RWMHSampling(),
    obs_sample = 1.0,
    init_params = 3.0,
    step = 0.25,
    rng = Random.GLOBAL_RNG,
)
    obs_sample = reshape(collect(obs_sample), 1) # scalar or Vector -> Vector
    init_params = reshape(collect(init_params), 1) # scalar or Vector -> Vector
    mcmc = MCMCWrapper(mcmc_alg, obs_sample, prior, em; init_params = init_params)

    # First let's run a short chain to determine a good step size
    new_step = optimize_stepsize(mcmc; init_stepsize = step, N = 5000)

    # Now begin the actual MCMC, sample is multiply exported so we qualify
    chain = MCMC.sample(rng, mcmc, 100_000; stepsize = new_step, discard_initial = 1000)
    posterior_distribution = get_posterior(mcmc, chain)
    #post_mean = mean(posterior, dims=1)[1]
    posterior_mean = mean(posterior_distribution)

    return new_step, posterior_mean[1]
end

@testset "MarkovChainMonteCarlo" begin
    obs_sample = [1.0]
    prior = test_prior()
    y, σ2_y, iopairs, rng = test_data(; rng_seed = 42, n = 40, var_y = 0.05)
    mcmc_params =
        Dict(:mcmc_alg => RWMHSampling(), :obs_sample => obs_sample, :init_params => [3.0], :step => 0.5, :rng => rng)

    @testset "Constructor: standardize" begin
        em = test_gp_1(y, σ2_y, iopairs)
        test_obs = MarkovChainMonteCarlo.to_decorrelated(obs_sample, em)
        # The MCMC stored a SVD-transformed sample,
        # 1.0/sqrt(0.05) * obs_sample ≈ 4.472
        @test isapprox(test_obs, (obs_sample ./ sqrt(σ2_y[1, 1])); atol = 1e-2)
    end

    @testset "MV priors" begin
        # 10D dist with 1 name, just build the wrapper for test
        prior_mv = test_prior_mv()
        y_mv, σ2_y_mv, iopairs_mv, rng_mv = test_data(input_dim = 10)
        init_params = repeat([0.0], 10)
        obs_sample = obs_sample # scalar or Vector -> Vector
        em_mv = test_gp_mv(y_mv, σ2_y_mv, iopairs_mv)
        mcmc = MCMCWrapper(RWMHSampling(), obs_sample, prior_mv, em_mv; init_params = init_params)

    end

    @testset "Sine GP & RW Metropolis" begin
        em_1 = test_gp_1(y, σ2_y, iopairs)
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params...)
        @test isapprox(new_step, 0.5; atol = 0.5)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        # now test SVD normalization
        norm_factor = 10.0
        norm_factor = fill(norm_factor, size(y[:, 1])) # must be size of output dim
        em_2 = test_gp_2(y, σ2_y, iopairs; norm_factor = norm_factor)
        _, posterior_mean_2 = mcmc_test_template(prior, σ2_y, em_2; mcmc_params...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
    end

    @testset "Sine GP & pCN" begin
        mcmc_params = Dict(
            :mcmc_alg => pCNMHSampling(),
            :obs_sample => obs_sample,
            :init_params => [3.0],
            :step => 0.5,
            :rng => rng,
        )

        em_1 = test_gp_1(y, σ2_y, iopairs)
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params...)
        @test isapprox(new_step, 0.25; atol = 0.25)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        # now test SVD normalization
        norm_factor = 10.0
        norm_factor = fill(norm_factor, size(y[:, 1])) # must be size of output dim
        em_2 = test_gp_2(y, σ2_y, iopairs; norm_factor = norm_factor)
        _, posterior_mean_2 = mcmc_test_template(prior, σ2_y, em_2; mcmc_params...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
    end
end
