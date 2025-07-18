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
using CalibrateEmulateSample.Utilities

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
function test_gp_mv(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    gp = GaussianProcess(gppackage; noise_learn = true, prediction_type = pred_type)
    em = Emulator(gp, iopairs; output_structure_matrix = σ2_y)
    Emulators.optimize_hyperparameters!(em)
    return em
end

function test_gp_1(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))
    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    em = Emulator(gp, iopairs; output_structure_matrix = σ2_y, encoder_schedule = [])
    Emulators.optimize_hyperparameters!(em)
    return em
end

function test_gp_and_agp_1(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))
    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    em = Emulator(gp, iopairs; output_structure_matrix = σ2_y, encoder_schedule = [])
    Emulators.optimize_hyperparameters!(em)

    # now make agp from gp
    agp = GaussianProcess(AGPJL(); noise_learn = true, prediction_type = pred_type)

    gp_opt_params = Emulators.get_params(gp)
    gp_opt_param_names = get_param_names(gp)

    kernel_params = [
        Dict(
            "log_rbf_len" => model_params[1:(end - 2)],
            "log_std_sqexp" => model_params[end - 1],
            "log_std_noise" => model_params[end],
        ) for model_params in gp_opt_params
    ]

    em_agp =
        Emulator(agp, iopairs, output_structure_matrix = σ2_y, encoder_schedule = [], kernel_params = kernel_params)

    return em, em_agp
end


function test_gp_2(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))
    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    retain_var = 0.95
    encoder_schedule = [(quartile_scale(), "out"), (decorrelate_structure_mat(retain_var = retain_var), "out")]
    em = Emulator(gp, iopairs; output_structure_matrix = σ2_y, encoder_schedule = encoder_schedule)
    Emulators.optimize_hyperparameters!(em)

    return em
end

function mcmc_test_template(
    prior::ParameterDistribution,
    σ2_y,
    em::Emulator;
    mcmc_alg = RWMHSampling(),
    obs_sample = [1.0],
    init_params = 3.0,
    step = 0.25,
    rng = Random.GLOBAL_RNG,
    target_acc = 0.25,
)
    if !isa(obs_sample, AbstractVecOrMat)
        obs_sample = reshape(collect(obs_sample), 1) # scalar -> Vector 
    end

    init_params = reshape(collect(init_params), 1) # scalar or Vector -> Vector
    mcmc = MCMCWrapper(mcmc_alg, obs_sample, prior, em; init_params = init_params)
    # First let's run a short chain to determine a good step size
    new_step = optimize_stepsize(mcmc; init_stepsize = step, N = 5000, target_acc = target_acc)

    # Now begin the actual MCMC, sample is multiply exported so we qualify
    chain = MCMC.sample(rng, mcmc, 50_000; stepsize = new_step, discard_initial = 10000)
    posterior_distribution = get_posterior(mcmc, chain)
    #post_mean = mean(posterior, dims=1)[1]
    posterior_mean = mean(posterior_distribution)

    return new_step, posterior_mean[1], chain
end

@testset "MarkovChainMonteCarlo" begin
    obs_sample = [1.0]
    prior = test_prior()
    y, σ2_y, iopairs, rng = test_data(; rng_seed = 42, n = 40, var_y = 0.05)
    mcmc_params =
        Dict(:mcmc_alg => RWMHSampling(), :obs_sample => obs_sample, :init_params => [3.0], :step => 0.5, :rng => rng)


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
        em_1, em_1b = test_gp_and_agp_1(y, σ2_y, iopairs)
        new_step_1, posterior_mean_1, chain_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params...)
        esjd1 = esjd(chain_1)
        @info "ESJD = $esjd1"
        @test isapprox(new_step_1, 0.5; atol = 0.5)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        # test the agp setup on without derivatives
        new_step_1b, posterior_mean_1b, chain_1b = mcmc_test_template(prior, σ2_y, em_1b; mcmc_params...)
        @test isapprox(new_step_1b, new_step_1; atol = 1e-1)
        tol_small = 5e-2
        @test isapprox(posterior_mean_1b, posterior_mean_1; atol = tol_small)
        esjd1b = esjd(chain_1b)
        @info "ESJD = $esjd1b"
        @test all(isapprox.(esjd1, esjd1b, rtol = 0.2))

        # now test SVD normalization
        em_2 = test_gp_2(y, σ2_y, iopairs)
        _, posterior_mean_2, chain_2 = mcmc_test_template(prior, σ2_y, em_2; mcmc_params...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
        # test diagnostic functions on the chain
        esjd2 = esjd(chain_2)
        @info "ESJD = $esjd2"
        # approx [0.04190683285347798, 0.1685296224916364, 0.4129400000002722]
        @test all(isapprox.(esjd1, esjd2, rtol = 0.2))

        # test with many slightly different samples
        # as vec of vec
        obs_sample2 = [obs_sample + 0.01 * randn(length(obs_sample)) for i in 1:100]
        mcmc_params2 = mcmc_params
        mcmc_params2[:obs_sample] = obs_sample2
        em_1 = test_gp_1(y, σ2_y, iopairs)
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params2...)
        @test isapprox(new_step, 0.5; atol = 0.5)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        # as column matrix
        obs_sample2mat = reduce(hcat, obs_sample2)
        mcmc_params2mat = mcmc_params
        mcmc_params2mat[:obs_sample] = obs_sample2mat
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params2mat...)
        @test isapprox(new_step, 0.5; atol = 0.5)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)


        # test with int data
        obs_sample3 = [1]
        mcmc_params3 = mcmc_params
        mcmc_params3[:obs_sample] = obs_sample3
        em_1 = test_gp_1(y, σ2_y, iopairs)
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params3...)
        @test isapprox(new_step, 0.5; atol = 0.5)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)


    end

    @testset "Sine GP & pCN" begin
        mcmc_params = Dict(
            :mcmc_alg => pCNMHSampling(),
            :obs_sample => obs_sample,
            :init_params => [3.0],
            :step => 0.5,
            :rng => rng,
        )

        em_1, em_1b = test_gp_and_agp_1(y, σ2_y, iopairs)
        new_step_1, posterior_mean_1, chain_1 = mcmc_test_template(prior, σ2_y, em_1; mcmc_params...)
        esjd1 = esjd(chain_1)
        @info "ESJD = $esjd1"
        @test isapprox(new_step_1, 0.75; atol = 0.6)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, π / 2; atol = 4e-1)

        # test the agp setup on without derivatives
        new_step_1b, posterior_mean_1b, chain_1b = mcmc_test_template(prior, σ2_y, em_1b; mcmc_params...)
        @test isapprox(new_step_1b, new_step_1; atol = 1e-1)
        tol_small = 5e-2
        @test isapprox(posterior_mean_1b, posterior_mean_1; atol = tol_small)
        esjd1b = esjd(chain_1b)
        @info "ESJD = $esjd1b"
        @test all(isapprox.(esjd1, esjd1b, rtol = 0.2))

        # now test SVD normalization
        em_2 = test_gp_2(y, σ2_y, iopairs)
        _, posterior_mean_2, chain_2 = mcmc_test_template(prior, σ2_y, em_2; mcmc_params...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)

        esjd2 = esjd(chain_2)
        @info "ESJD = $esjd2"
        # approx [0.03470825350663073, 0.161606734823579, 0.38970000000024896]

        @test all(isapprox.(esjd1, esjd2, rtol = 0.2))

    end




    mcmc_params = Dict(:obs_sample => obs_sample, :init_params => [3.0], :step => 0.25, :rng => rng, :target_acc => 0.6) # the target is usually higher in grad-based MCMC

    em_1, em_1b = test_gp_and_agp_1(y, σ2_y, iopairs)
    # em_1 cannot be used here

    mcmc_algs = [
        RWMHSampling(), # sanity-check
        BarkerSampling(), # ForwardDiffProtocol by default
        BarkerSampling{ReverseDiffProtocol}(), # scales to high dim better, but slow.
    ]

    bad_mcmc_alg = BarkerSampling{GradFreeProtocol}()
    @test_throws ArgumentError mcmc_test_template(prior, σ2_y, em_1; mcmc_alg = bad_mcmc_alg, mcmc_params...)


    # GPJL doesnt support ForwardDiff
    @test_throws ErrorException mcmc_test_template(prior, σ2_y, em_1; mcmc_alg = mcmc_algs[2], mcmc_params...)

    for alg in mcmc_algs
        @info "testing algorithm: $(typeof(alg))"
        new_step, posterior_mean, chain = mcmc_test_template(prior, σ2_y, em_1b; mcmc_alg = alg, mcmc_params...)
        esjd_tmp = esjd(chain)
        @info "ESJD = $esjd_tmp"
        @info posterior_mean
        @testset "Sine GP & ForwardDiff variant:$(typeof(alg))" begin
            @test isapprox(posterior_mean, π / 2; atol = 4e-1)
        end

    end

end
