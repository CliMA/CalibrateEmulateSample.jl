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

# range 0->2, with lengthscale of transition 0.5, and y=1 at x=2
G(x) = 5 * (tanh.((x .- 2) ./ 0.5) .+ 1)

function test_data(prior; rng_seed = 41, n = 80, var_y = 0.05, rest...)
    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)
    # We need a GaussianProcess to run MarkovChainMonteCarlo, so let's reconstruct the one 
    # that's tested in test/GaussianProcesses/runtests.jl Case 1
    x = 5 * rand(rng, Float64, (1, n))                 # predictors/features: 1 × n
    σ2_y = var_y * I #reshape([var_y], 1, 1)
    y = G(x) + sqrt(σ2_y.λ) * randn(rng, 1, n) # predictands/targets: 1 × n

    return y,
    σ2_y,
    PairedDataContainer(transform_constrained_to_unconstrained(prior, x), y, data_are_columns = true),
    rng
end

function validate_emulator(em, data_name, prior, test_data_kwargs; exp_name = "test")
    tpk = merge(test_data_kwargs, (; rng_seed = 235412))

    if data_name == "test_data"
        _, _, validation_data, _ = test_data(prior; test_data_kwargs...)
    elseif data_name == "test_data_2d"
        _, _, validation_data, _ = test_data_2d(prior; test_data_kwargs...)
    elseif data_name == "test_data_mv"
        _, _, validation_data, _ = test_data_mv(prior; test_data_kwargs...)
    end


    μ_valid, σ2_valid = Emulators.predict(em, get_inputs(validation_data), transform_to_real = true)
    rmse_valid = norm(μ_valid - get_outputs(validation_data)) ./ sqrt(size(get_outputs(validation_data), 2))
    avg_σ2_valid = mean([norm(ss) for ss in σ2_valid])
    @info "$(exp_name) \n per-point - Emulator RMSE = $rmse_valid \n Emulator STD = $(sqrt(avg_σ2_valid))\n (These numbers should be similar sized to indicate no overfitting)"

    if TEST_PLOT_OUTPUT
        if data_name == "test_data" #(1D->1D)
            sorted_idx = sortperm(get_inputs(validation_data)[1, :])
            sorted_in = get_inputs(validation_data)[:, sorted_idx][:]
            sorted_out = get_outputs(validation_data)[:, sorted_idx][:]
            μ_plot = μ_valid[:, sorted_idx][:]
            σ2_plot = σ2_valid[:, sorted_idx][:]

            plot(sorted_in, sorted_out, label = "data", title = exp_name)
            plot!(sorted_in, μ_plot, yerror = 2 * sqrt.(σ2_plot), label = "mean, 95% interval")

            savefig(joinpath(@__DIR__, "validate_emulator_$(exp_name).png"))
        end
    end
end

function test_data_2d(prior; rng_seed = 4141, n = 100, cov_y = 0.05 * [[0.5, 0.2] [0.2, 0.5]], rest...)
    rng = Random.MersenneTwister(rng_seed)
    n = 100 # number of training points

    input_dim = 2   # input dim
    output_dim = 2   # output dim
    X = 5 * rand(rng, Float64, (input_dim, n)) # bounds from prior on input space are [-1,6]

    # G(x1, x2)
    g1x = sin.(X[1, :]) .+ cos.(X[2, :])
    g2x = sin.(X[1, :]) .- cos.(X[2, :])
    gx = zeros(2, n)
    gx[1, :] = g1x
    gx[2, :] = g2x

    # Add noise η
    μ = zeros(output_dim)
    noise_samples = rand(MvNormal(μ, cov_y), n)

    # y = G(x) + η
    Y = gx .+ noise_samples

    iopairs = PairedDataContainer(transform_constrained_to_unconstrained(prior, X), Y, data_are_columns = true)
    return Y, cov_y, iopairs, rng
end

function test_prior()
    return constrained_gaussian("u", 2.0, 1.0, -1.0, 6.0)
end


function test_prior_mv(input_dim = 10)
    ### Define prior
    return constrained_gaussian("u_mv", 2.0, 1.0, -1.0, 6.0, repeats = input_dim)
end

test_prior_2d() = test_prior_mv(2)

function test_data_mv(prior; rng_seed = 41, n = 500, var_y = 0.01, input_dim = 4, rest...)

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(rng_seed)
    # number of training points
    x = 5.0 * rand(rng, Float64, (input_dim, n))                 # predictors/features: 1 × n
    σ2_y = var_y * I #reshape([var_y], 1, 1)
    y = reshape(G([norm(xx) / input_dim for xx in eachcol(x)]), 1, n) + rand(rng, MvNormal(zeros(1), σ2_y), n) # predictands/targets: 1 × n

    return y,
    σ2_y,
    PairedDataContainer(transform_constrained_to_unconstrained(prior, x), y, data_are_columns = true),
    rng
end

function test_gp_mv(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    gp = GaussianProcess(gppackage; noise_learn = false, prediction_type = pred_type)
    em = Emulator(gp, iopairs; encoder_kwargs = (; obs_noise_cov = σ2_y))
    Emulators.optimize_hyperparameters!(em)
    return em
end

function test_gp_1(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    #GPkernel = SE(log(1.0), log(1.0)) #kernel = GPkernel
    gp = GaussianProcess(gppackage; noise_learn = false, prediction_type = pred_type)
    em = Emulator(gp, iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = σ2_y))
    Emulators.optimize_hyperparameters!(em)
    return em
end

function test_gp_and_agp_1(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    #GPkernel = SE(log(1.0), log(1.0)) #kernel = GPkernel
    noise_learn_gp = false
    gp = GaussianProcess(gppackage; noise_learn = false, prediction_type = pred_type)
    em = Emulator(gp, iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = σ2_y))
    Emulators.optimize_hyperparameters!(em)

    # now make agp from gp
    agp = GaussianProcess(AGPJL(); noise_learn = false, prediction_type = pred_type) # NB it won't learn the noise here...

    gp_opt_params = Emulators.get_params(gp)
    gp_opt_param_names = get_param_names(gp)

    if noise_learn_gp
        kernel_params = [
            Dict(
                "log_rbf_len" => model_params[1:(end - 2)],
                "log_std_sqexp" => model_params[end - 1],
                "log_std_noise" => model_params[end],
            ) for model_params in gp_opt_params
        ]
    else
        kernel_params = [
            Dict(
                "log_rbf_len" => model_params[1:(end - 1)],
                "log_std_sqexp" => model_params[end],
                "log_std_noise" => -10, # large+neg. will make this term ~0. (noise will come from the regularization matrix in encoder).
            ) for model_params in gp_opt_params
        ]
    end

    em_agp = Emulator(
        agp,
        iopairs;
        encoder_schedule = [],
        encoder_kwargs = (; obs_noise_cov = σ2_y),
        kernel_params = kernel_params,
    )

    return em, em_agp
end


function test_gp_2(y, σ2_y, iopairs::PairedDataContainer)
    gppackage = GPJL()
    pred_type = YType()
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))
    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = false, prediction_type = pred_type)
    retain_var = 0.95
    encoder_schedule = [(quartile_scale(), "out"), (decorrelate_structure_mat(retain_var = retain_var), "out")]
    em = Emulator(gp, iopairs; encoder_schedule = encoder_schedule, encoder_kwargs = (; obs_noise_cov = σ2_y))
    Emulators.optimize_hyperparameters!(em)

    return em
end

function test_srfi(y, σ2_y, iopairs::PairedDataContainer)
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    input_dim = size(get_inputs(iopairs), 1)
    n_features = 100
    kernel_structure = SeparableKernel(DiagonalFactor(), OneDimFactor())
    srfi = ScalarRandomFeatureInterface(n_features, input_dim, kernel_structure = kernel_structure)

    em = Emulator(srfi, iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = σ2_y))

    Emulators.optimize_hyperparameters!(em)

    return em
end

function test_vrfi(y, σ2_y, iopairs::PairedDataContainer)
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    input_dim = size(get_inputs(iopairs), 1)
    output_dim = size(get_outputs(iopairs), 1)
    n_features = 100
    kernel_structure = NonseparableKernel(LowRankFactor(2))
    vrfi = VectorRandomFeatureInterface(n_features, input_dim, output_dim, kernel_structure = kernel_structure)

    em = Emulator(vrfi, iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = σ2_y))
    Emulators.optimize_hyperparameters!(em)

    return em
end


function mcmc_test_template(
    prior::ParameterDistribution,
    σ2_y,
    em::Emulator;
    exp_name = "test",
    mcmc_alg = RWMHSampling(),
    obs_sample = [1.0],
    init_params = transform_constrained_to_unconstrained(prior, [2.0]),
    step = 0.25,
    rng = Random.GLOBAL_RNG,
    target_acc = 0.25,
    return_samples = false,
)
    if !isa(obs_sample, AbstractVecOrMat)
        obs_sample = reshape(collect(obs_sample), 1) # scalar -> Vector 
    end
    init_params = vec(collect(init_params)) # scalar or Vector -> Vector

    mcmc = MCMCWrapper(mcmc_alg, obs_sample, prior, em; init_params = init_params)
    # First let's run a short chain to determine a good step size
    new_step = optimize_stepsize(mcmc; init_stepsize = step, N = 5000, target_acc = target_acc)

    # Now begin the actual MCMC, sample is multiply exported so we qualify
    chain = MCMC.sample(rng, mcmc, 50_000; stepsize = new_step, discard_initial = 10000)
    posterior_distribution = get_posterior(mcmc, chain)
    if TEST_PLOT_OUTPUT
        # plot:
        pp = plot(prior, title = exp_name, color = :grey)
        plot!(pp, posterior_distribution, color = :blue)
        savefig(pp, joinpath(@__DIR__, "posterior_$(exp_name).png"))
    end


    if return_samples # return sanples not mean
        constrained_posterior_samples =
            transform_unconstrained_to_constrained(prior, get_distribution(posterior_distribution)) # still a Dict (param_name => samples)
        return new_step, constrained_posterior_samples, chain
    else
        constrained_posterior_mean = transform_unconstrained_to_constrained(prior, mean(posterior_distribution))
        if length(constrained_posterior_mean) == 1
            return new_step, constrained_posterior_mean[1], chain
        else
            return new_step, constrained_posterior_mean, chain
        end
    end

end

@testset "MarkovChainMonteCarlo" begin

    # Problem and emulator setups: then MCMC tests come after

    # [1.] 1D -> 1D
    # setup
    obs_sample = [5.0]
    mle = 2.0
    prior = test_prior()
    test_data_kwargs = (; rng_seed = 42, n = 80, var_y = 0.05)
    y, σ2_y, iopairs, rng = test_data(prior; test_data_kwargs...) # iopairs unconstrained inputs

    # mcmc setup
    mcmc_params = Dict(
        :mcmc_alg => RWMHSampling(),
        :obs_sample => obs_sample,
        :init_params => transform_constrained_to_unconstrained(prior, [2.0]),
        :step => 0.25,
        :rng => rng,
        :target_acc => 0.25,
    )

    mcmc_params_pcn = deepcopy(mcmc_params)
    mcmc_params_pcn[:mcmc_alg] = pCNMHSampling()
    mcmc_params_pcn[:target_acc] = 0.6
    mcmc_params_pcn[:step] = 0.025

    # build and validate the emulators
    em_1, em_1b = test_gp_and_agp_1(y, σ2_y, iopairs)
    em_2 = test_gp_2(y, σ2_y, iopairs)
    em_s = test_srfi(y, σ2_y, iopairs)

    validate_emulator(em_s, "test_data", prior, test_data_kwargs, exp_name = "srfi_1d")
    validate_emulator(em_1, "test_data", prior, test_data_kwargs, exp_name = "gpjl_1d")
    validate_emulator(em_1b, "test_data", prior, test_data_kwargs, exp_name = "agpjl_1d")
    validate_emulator(em_2, "test_data", prior, test_data_kwargs, exp_name = "gpjl-svd_1d")

    # [2.] 10D -> 1D
    # setup
    obs_sample_mv = [5.0]
    input_dim = 5
    mle_norm_mv = 2.0 * input_dim
    prior_mv = test_prior_mv(input_dim)
    test_data_mv_kwargs = (; input_dim = input_dim)
    y_mv, σ2_y_mv, iopairs_mv, rng_mv = test_data_mv(prior_mv; test_data_mv_kwargs...) # iopairs unconstrained inputs

    # build and validate the emulators
    em_mv = test_gp_mv(y_mv, σ2_y_mv, iopairs_mv)
    validate_emulator(em_mv, "test_data_mv", prior_mv, test_data_mv_kwargs, exp_name = "gp_mv")

    # mcmc setup
    mcmc_params_mv = Dict(
        :mcmc_alg => RWMHSampling(),
        :obs_sample => obs_sample_mv,
        :init_params => transform_constrained_to_unconstrained(prior_mv, repeat([2.1], input_dim)),
        :step => 0.25,
        :rng => rng_mv,
    )

    mcmc_params_mv_pcn = deepcopy(mcmc_params_mv)
    mcmc_params_mv_pcn[:mcmc_alg] = pCNMHSampling()
    mcmc_params_mv_pcn[:target_acc] = 0.6
    mcmc_params_mv_pcn[:step] = 0.025

    # [3.] 2D -> 2D

    # setup
    obs_sample_2d = [1.0, 1.0]
    mle_2d = [π / 2, π / 2]
    prior_2d = test_prior_2d()
    test_data_2d_kwargs = (; rng_seed = 4141, n = 100, cov_y = 0.05 * [[0.5, 0.2] [0.2, 0.5]])
    y_2d, σ2_y_2d, iopairs_2d, rng_2d = test_data_2d(prior_2d; test_data_2d_kwargs...)
    em_v = test_vrfi(y_2d, σ2_y_2d, iopairs_2d)
    validate_emulator(em_v, "test_data_2d", prior_2d, (; test_data_2d_kwargs))

    mcmc_params_2d = Dict(
        :mcmc_alg => RWMHSampling(),
        :obs_sample => obs_sample_2d,
        :init_params => transform_constrained_to_unconstrained(prior_2d, [2.0, 2.0]),
        :step => 0.025,
        :rng => rng_2d,
    )

    @testset "1D-1D GP/RF & RW Metropolis" begin

        # test various MCMC methods
        new_step_1, posterior_mean_1, chain_1 =
            mcmc_test_template(prior, σ2_y, em_1; exp_name = "gpjl_1d", mcmc_params...)
        esjd1 = esjd(chain_1)
        @info "ESJD [GPJL,RW] = $esjd1"
        @test isapprox(new_step_1, 0.125; atol = 0.125)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @test isapprox(posterior_mean_1, mle; atol = 1e-1)
        @info "Posterior mean: $(posterior_mean_1) ≈ $(mle)"

        # test the agp setup on without derivatives
        new_step_1b, posterior_mean_1b, chain_1b =
            mcmc_test_template(prior, σ2_y, em_1b; exp_name = "agpjl_1d", mcmc_params...)
        @test isapprox(new_step_1b, new_step_1; atol = 0.125)
        tol_small = 0.2
        esjd1b = esjd(chain_1b)
        @info "ESJD [AGPJL,RW] = $esjd1b"
        @test all(isapprox.(esjd1, esjd1b, rtol = 0.2))
        @test isapprox(posterior_mean_1b, posterior_mean_1; atol = tol_small)
        @info "Posterior mean: $(posterior_mean_1b) ≈ $(mle)"

        # test with random features

        new_step_s, posterior_mean_s, chain_s =
            mcmc_test_template(prior, σ2_y, em_s; exp_name = "srfi_1d", mcmc_params...)
        esjds = esjd(chain_s)
        @info "ESJD [SRFI,RW] = $esjds"
        # approx [0.0002, 0.26, 0.4]"
        @test all(isapprox.(esjd1, esjds, rtol = 0.5))
        @info "Posterior mean: $(posterior_mean_s) ≈ $(mle)"
        @test isapprox(posterior_mean_s, mle; atol = 2e-1)

        # now test SVD normalization
        _, posterior_mean_2, chain_2 = mcmc_test_template(prior, σ2_y, em_2; exp_name = "gpjl-svd_1d", mcmc_params...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        # test diagnostic functions on the chain
        esjd2 = esjd(chain_2)
        @info "ESJD (-with-svd)= $esjd2"
        # approx [0.00015, 0.26, 0.35]
        @test all(isapprox.(esjd1, esjd2, rtol = 0.5))
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
        @info "Posterior mean: $(posterior_mean_2) ≈ $(mle)"

        # test with many slightly different samples
        # as vec of vec
        obs_sample2 = [obs_sample + 0.01 * randn(length(obs_sample)) for i in 1:100]
        mcmc_params2 = deepcopy(mcmc_params)
        mcmc_params2[:obs_sample] = obs_sample2
        mcmc_params2[:step] = 0.025 # less uncertainty -> smaller step
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; exp_name = "gpjl-samples", mcmc_params2...)
        @test isapprox(new_step, 0.025; atol = 0.025)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        esjd2 = esjd(chain_2)
        @info "ESJD (-many-samples)= $esjd2"
        @info "Posterior mean: $(posterior_mean_1) ≈ $(mle)"
        @test isapprox(posterior_mean_1, mle; atol = 2e-1)


        # as column matrix
        obs_sample2mat = reduce(hcat, obs_sample2)
        mcmc_params2mat = deepcopy(mcmc_params)
        mcmc_params2mat[:obs_sample] = obs_sample2mat
        new_step, posterior_mean_1 =
            mcmc_test_template(prior, σ2_y, em_1; exp_name = "gpjl-samples-mat", mcmc_params2mat...)
        @test isapprox(new_step, 0.025; atol = 0.025)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @info "Posterior mean: $(posterior_mean_1) ≈ $(mle)"
        @test isapprox(posterior_mean_1, mle; atol = 2e-1)

        # test with integer data
        obs_sample3 = [4]
        mcmc_params3 = deepcopy(mcmc_params)
        mcmc_params3[:obs_sample] = obs_sample3
        new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; exp_name = "gpjl_int_1d", mcmc_params3...)
        @test isapprox(new_step, 0.5; atol = 0.5)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @info "Posterior mean: $(posterior_mean_1) ≈ $(mle)"
        @test isapprox(posterior_mean_1, mle; atol = 2e-1)


    end


    @testset "2D-2D RF & RW" begin

        new_step_v, posterior_mean_v, chain_v = mcmc_test_template(prior_2d, σ2_y_2d, em_v; mcmc_params_2d...)
        @test isapprox(new_step_v, 0.5; atol = 0.5)
        esjdv = esjd(chain_v)
        @info "ESJD [VRFI, 2d]= $esjdv"
        # [0.025, 0.003, 0.2, 0.27]
        @info "Posterior mean: $(posterior_mean_v) ≈ $(mle_2d)"
        @test all(isapprox.(posterior_mean_v - mle_2d, 0.0; atol = 4e-1))

    end


    @testset "1D-1D pCN" begin

        new_step_1, posterior_mean_1, chain_1 =
            mcmc_test_template(prior, σ2_y, em_1; exp_name = "gpjl_pcn", mcmc_params_pcn...)
        esjd1 = esjd(chain_1)
        @info "ESJD [GPJL,pCN] = $esjd1"
        @test isapprox(new_step_1, 0.125; atol = 0.125)
        # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
        @info "Posterior mean: $(posterior_mean_1) ≈ $mle"
        @test isapprox(posterior_mean_1, mle; atol = 2e-1)

        # test the agp setup on without derivatives
        new_step_1b, posterior_mean_1b, chain_1b =
            mcmc_test_template(prior, σ2_y, em_1b; exp_name = "agpjl_pcn", mcmc_params_pcn...)
        @test isapprox(new_step_1b, new_step_1; atol = 0.125)
        tol_small = 1e-1
        @test isapprox(posterior_mean_1b, posterior_mean_1; atol = tol_small)
        esjd1b = esjd(chain_1b)
        @info "ESJD [AGPJL,pCN] = $esjd1b"
        @info "Posterior mean: $(posterior_mean_1b) ≈ $mle"
        @test all(isapprox.(esjd1, esjd1b, rtol = 0.2))

        # test with random features
        new_step_s, posterior_mean_s, chain_s =
            mcmc_test_template(prior, σ2_y, em_s; exp_name = "srfi_1d_pcn", mcmc_params_pcn...)
        @test isapprox(posterior_mean_s, mle; atol = 2e-1)
        esjds = esjd(chain_s)
        @info "ESJD [SRFI,pCN] = $esjds"
        # approx [0.0002, 0.26, 0.4]"
        @info "Posterior mean: $(posterior_mean_s) ≈ $mle"
        @test all(isapprox.(esjd1, esjds, rtol = 0.5))

        # now test SVD normalization
        _, posterior_mean_2, chain_2 = mcmc_test_template(prior, σ2_y, em_2; mcmc_params_pcn...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)

        esjd2 = esjd(chain_2)
        @info "ESJD = $esjd2"
        # approx [0.03470825350663073, 0.161606734823579, 0.38970000000024896]

        @test all(isapprox.(esjd1, esjd2, rtol = 0.2))

    end

    @testset "ND-1D" begin
        @info "Input dimension: $(input_dim)"
        for params in (mcmc_params_mv, mcmc_params_mv_pcn)
            # 10D dist with 1 name, just build the wrapper for test
            new_step_mv, constrained_posterior_samples_mv, chain_mv =
                mcmc_test_template(prior_mv, σ2_y_mv, em_mv; exp_name = "gpjl_10d", return_samples = true, params...)
            esjd_mv = esjd(chain_mv)
            @info "ESJD (ND-1D) [GPJL,RW] = $esjd_mv"
            # as function performs f(norm(x)), posterior should have good properties of the mean(norm(x))
            samples_mat_mv = reduce(vcat, [v for v in values(constrained_posterior_samples_mv)])
            @info samples_mat_mv[:, (end - 20):end]
            norm_samples_mv = norm.([c for c in eachcol(samples_mat_mv)])
            posterior_norm_mean_mv = mean(norm_samples_mv)
            G_mean_mv = mean(G(norm_samples_mv / input_dim))
            @info "Mean of posterior sample norm: $(posterior_norm_mean_mv) ≈ $(mle_norm_mv)"
            @info "Mean of G sample norm: $(G_mean_mv) ≈ $(obs_sample_mv)"
        end
    end

    @testset "Autodiff MCMC variants" begin
        mcmc_algs = [
            RWMHSampling(), # sanity-check
            BarkerSampling(), # ForwardDiffProtocol by default
            BarkerSampling{ReverseDiffProtocol}(), # scales to high dim better, but slow.
        ]

        bad_mcmc_alg = BarkerSampling{GradFreeProtocol}()
        bad_mcmc_params = deepcopy(mcmc_params)
        bad_mcmc_params[:mcmc_alg] = bad_mcmc_alg

        @test_throws ArgumentError mcmc_test_template(prior, σ2_y, em_1; bad_mcmc_params...)

        # GPJL doesnt support ForwardDiff
        bad_mcmc_params = deepcopy(mcmc_params)
        bad_mcmc_params[:mcmc_alg] = mcmc_algs[2]
        @test_throws ErrorException mcmc_test_template(prior, σ2_y, em_1; bad_mcmc_params...)

        for alg in mcmc_algs
            mcmc_params_ad = deepcopy(mcmc_params)
            mcmc_params_ad[:mcmc_alg] = alg
            mcmc_params_ad[:target_acc] = 0.6 # should be > 0.5

            @info "testing algorithm: $(typeof(alg))"
            new_step, posterior_mean, chain = mcmc_test_template(prior, σ2_y, em_1b; mcmc_alg = alg, mcmc_params_ad...)
            esjd_tmp = esjd(chain)
            @info "ESJD = $esjd_tmp"
            @info "Posterior mean: $(posterior_mean) ≈ $mle"
            @testset "Sine GP & ForwardDiff variant:$(typeof(alg))" begin
                @test isapprox(posterior_mean, mle; atol = 1e-1)
            end
        end
    end

end
