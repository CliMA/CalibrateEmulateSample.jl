using Random
using LinearAlgebra
using Distributions
using GaussianProcesses
using Test
using AdvancedMH
using AbstractMCMC
using MCMCChains

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.MarkovChainMonteCarlo
const MCMC = MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
const PD = ParameterDistributions
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Utilities

# range 0->2, with lengthscale of transition 0.5, and y=1 at x=2
G(x) = 5 * (tanh.((x .- 2) ./ 0.5) .+ 1)

# A minimal MHSampler that deliberately does not implement log_transition_density, used to check
# that the shared interface errors loudly instead of silently falling back to a wrong value.
struct _DummyMHSampler <: AdvancedMH.MHSampler end

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


    μ_valid, σ2_valid = Emulators.predict(em, get_inputs(validation_data); add_obs_noise_cov = true)
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
    # scaled by input_dim, so that the prior/training inputs remain reasonable in different input dimensions.
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
    if isa(obs_sample, Real)
        obs_sample = reshape(collect(obs_sample), 1) # scalar -> Vector 
    end
    init_params = vec(collect(init_params)) # scalar or Vector -> Vector

    mcmc = MCMCWrapper(mcmc_alg, obs_sample, prior, em) # without ICs

    enc_sch = get_encoder_schedule(em)
    mp = ndims(prior) > 1 ? mean(prior) : [mean(prior)]
    encoded_ics = encode_data(enc_sch, reshape(mp, :, 1), "in")
    @test all(isapprox.(getfield(get_sample_kwargs(mcmc), :initial_params), encoded_ics))

    mcmc = MCMCWrapper(mcmc_alg, obs_sample, prior, em; init_params = init_params)
    # First let's run a short chain to determine a good step size
    new_step = optimize_stepsize(mcmc; init_stepsize = step, N = 2000, target_acc = target_acc)

    # Now begin the actual MCMC, sample is multiply exported so we qualify
    chain = MCMC.sample(rng, mcmc, 20_000; stepsize = new_step, discard_initial = 2_000)
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
        :obs_sample => Observation(Dict("samples" => obs_sample, "covariances" => 1.0 * I, "names" => "test")),
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

    # [2.] 5D -> 1D
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
        @test all(isapprox.(esjd1, esjds, rtol = 0.75))
        @info "Posterior mean: $(posterior_mean_s) ≈ $(mle)"
        @test isapprox(posterior_mean_s, mle; atol = 2e-1)

        # now test SVD normalization
        _, posterior_mean_2, chain_2 = mcmc_test_template(prior, σ2_y, em_2; exp_name = "gpjl-svd_1d", mcmc_params...)
        # difference between mean_1 and mean_2 only from MCMC convergence
        # test diagnostic functions on the chain
        esjd2 = esjd(chain_2)
        @info "ESJD (-with-svd)= $esjd2"
        # approx [0.00015, 0.26, 0.35]
        @test all(isapprox.(esjd1, esjd2, rtol = 0.75))
        @test isapprox(posterior_mean_2, posterior_mean_1; atol = 0.1)
        @info "Posterior mean: $(posterior_mean_2) ≈ $(mle)"

        # test with many slightly different samples
        obs_sample_vecs = []
        exp_vec_names = []

        # as a vec of vec
        obs_sample2 = [obs_sample + 0.01 * randn(length(obs_sample)) for i in 1:100]
        push!(exp_vec_names, "gpjl-samples-mat")
        push!(obs_sample_vecs, obs_sample2)

        # as a column mat
        obs_sample2mat = reduce(hcat, obs_sample2)
        push!(obs_sample_vecs, obs_sample2mat)
        push!(exp_vec_names, "gpjl-samples")

        # as an observation series
        obs_vec = []
        for i in 1:100
            ob = Observation(
                Dict("samples" => obs_sample + 0.01 * randn(length(obs_sample)), "covariances" => I, "names" => "test"),
            )
            push!(obs_vec, ob)
        end
        obs_series = ObservationSeries(obs_vec)
        push!(obs_sample_vecs, obs_series)
        push!(exp_vec_names, "gpjl-samples-series")

        # run tests...
        for (os, exp_name) in zip(obs_sample_vecs, exp_vec_names)
            mcmc_params2 = deepcopy(mcmc_params)
            mcmc_params2[:obs_sample] = os
            mcmc_params2[:step] = 0.025 # less uncertainty -> smaller step
            new_step, posterior_mean_1 = mcmc_test_template(prior, σ2_y, em_1; exp_name = exp_name, mcmc_params2...)
            @test isapprox(new_step, 0.025; atol = 0.025)
            # difference between mean_1 and ground truth comes from MCMC convergence and GP sampling
            esjd2 = esjd(chain_2)
            @info "ESJD (vec:$(exp_name))= $esjd2"
            @info "Posterior mean: $(posterior_mean_1) ≈ $(mle)"
            @test isapprox(posterior_mean_1, mle; atol = 2e-1)
        end

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
        @test all(isapprox.(esjd1, esjds, rtol = 0.75))

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
            norm_samples_mv = norm.([c for c in eachcol(samples_mat_mv)])
            posterior_norm_mean_mv = mean(norm_samples_mv)
            G_mean_mv = mean(G(norm_samples_mv / input_dim))
            @info "Mean of posterior sample norm: $(posterior_norm_mean_mv) ≈ $(mle_norm_mv)"
            @info "Mean of G sample norm: $(G_mean_mv) ≈ $(obs_sample_mv[1])"
        end
    end


    @testset "Test the encode-decode for posterior samples" begin

        # build a mv problem in input space
        input_dim = 10
        n_samples = 5000
        prior_mv = test_prior_mv(input_dim)
        test_data_mv_kwargs = (; input_dim = input_dim)
        y_mv, σ2_y_mv, iopairs_mv, rng_mv = test_data_mv(prior_mv; test_data_mv_kwargs...) # iopairs unconstrained inputs

        # new data
        samples = 0.1 * collect(1:input_dim) .* randn(input_dim, n_samples)

        # lossless encoding
        lossless_sch = create_encoder_schedule((minmax_scale(), "in"))
        initialize_and_encode_with_schedule!(lossless_sch, iopairs_mv; prior_cov = cov(prior_mv))

        enc_samples = encode_data(lossless_sch, samples, "in")
        ni_scaling = 1.0 # don't scale the noise injected samples
        full_samples = decode_and_add_noise(lossless_sch, enc_samples, prior_mv, 1.0, ni_scaling) # 1.0 = no boosting, should be like decoding
        tol = 1e-12
        enc_factor = size(enc_samples, 1) * size(enc_samples, 2)
        dec_factor = size(full_samples, 1) * size(full_samples, 2)
        @test isapprox(norm(full_samples - decode_data(lossless_sch, enc_samples, "in")), 0; atol = tol * dec_factor)

        # lossy encoding
        lossy_sch = create_encoder_schedule((decorrelate_sample_cov(retain_var = 0.8), "in")) # lossy drops ~ 0.2 variance       
        initialize_and_encode_with_schedule!(lossy_sch, iopairs_mv; prior_cov = cov(prior_mv))

        enc_samples = encode_data(lossy_sch, samples, "in")
        # -> without noise injection
        dec_samples = decode_and_add_noise(lossy_sch, enc_samples, prior_mv, 0.25, ni_scaling) # should not boost (0.25>0.2)
        @test isapprox(norm(dec_samples - decode_data(lossy_sch, enc_samples, "in")), 0; atol = tol * dec_factor)
        @test isapprox(norm(encode_data(lossy_sch, dec_samples, "in") - enc_samples), 0; atol = tol * enc_factor)

        # -> with noise injection
        full_samples = decode_and_add_noise(lossy_sch, enc_samples, prior_mv, 0.15, ni_scaling) # should boost (0.15 < 0.2)

        # check re encoding samples gives same reduced samples (will be rough when lossy drops larger variance %)
        bigger_tol = 1e-6
        @test isapprox(
            norm(encode_data(lossy_sch, full_samples, "in") - enc_samples),
            0;
            atol = bigger_tol * enc_factor,
        )

        # check that the fill-in has distribution that matches C - KEC
        # compute deterinistic part
        E, b = get_encoder_from_schedule(lossy_sch, "in")
        E = Matrix(E)
        C = cov(prior_mv)
        m = reshape(mean(prior_mv), :, 1)
        ECEt = cholesky(Symmetric(E * C * E' + 1e-12I))
        K = C * E' * (ECEt \ I)
        det_part = m .+ K * (enc_samples .- (E * m .+ b))

        # asses the covariance of the remainder
        C_rem = cov(full_samples - det_part, dims = 2)
        @test isapprox(norm(C_rem - (C - K * E * C)), 0; atol = input_dim / sqrt(n_samples)) #  N is huge

        # check noise injector components:
        noise_injector = create_noise_injector(lossy_sch, prior_mv, 0.0, 0.5)
        @test isapprox(norm(K - noise_injector.K), 0; atol = tol * size(K, 1) * size(K, 2))
        @test isapprox(norm(E * m + b - noise_injector.enc_m), 0; atol = tol * size(noise_injector.enc_m, 1))
        @test isapprox(norm(m - noise_injector.m), 0; atol = tol * size(m, 1))
        @test isapprox(
            norm(cholesky(Symmetric(C_rem + 1e-12 * I)).L - noise_injector.L),
            0;
            atol = input_dim / sqrt(n_samples),
        )
        @test noise_injector.use_noise # check for noise_injector_threshold
        @test noise_injector.scaling == 0.5

        # check 1D
        input_dim = 1
        n_samples = 10
        prior_1d = constrained_gaussian("1d-check", 0, 1, -Inf, 5)
        in_data = PD.sample(prior_1d, n_samples)
        out_data = PD.sample(prior_1d, n_samples)
        io_pairs_1d = PairedDataContainer(in_data, out_data, data_are_columns = true)

        # lossless encoding
        lossless_sch = create_encoder_schedule((minmax_scale(), "in"))
        initialize_and_encode_with_schedule!(lossless_sch, io_pairs_1d; prior_cov = cov(prior_1d))

        noise_injector = create_noise_injector(lossless_sch, prior_1d, 0.0, 0.5)


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

        let thrown = @test_throws ArgumentError mcmc_test_template(prior, σ2_y, em_1; bad_mcmc_params...)
            @test contains(thrown.value.msg, "autodiff_gradient")
            @test contains(thrown.value.msg, "GradFreeProtocol")
            @test contains(thrown.value.msg, "ForwardDiffProtocol")
        end

        # GPJL doesnt support ForwardDiff
        bad_mcmc_params = deepcopy(mcmc_params)
        bad_mcmc_params[:mcmc_alg] = mcmc_algs[2]
        let thrown = @test_throws ArgumentError mcmc_test_template(prior, σ2_y, em_1; bad_mcmc_params...)
            @test contains(thrown.value.msg, "does not implement the required emulator interface")
        end

        for alg in mcmc_algs
            mcmc_params_ad = deepcopy(mcmc_params)
            mcmc_params_ad[:mcmc_alg] = alg
            # 0.4 sits comfortably inside the tight, low-variance region of Barker's
            # acceptance-vs-stepsize curve for this problem; 0.6 sits right at its edge (measured
            # ceiling ~0.48-0.50 here), which occasionally made optimize_stepsize unable to find a
            # valid stepsize at all.
            mcmc_params_ad[:target_acc] = 0.4

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

    @testset "optimize_stepsize: branch coverage" begin
        # optimize_stepsize's bracket-and-bisect search has several distinct code paths 
        mcmc = MCMCWrapper(RWMHSampling(), mcmc_params[:obs_sample], prior, em_1; init_params = vec(collect(mcmc_params[:init_params])))

        @testset "returns immediately if the initial stepsize is already within tolerance" begin
            step = optimize_stepsize(Random.MersenneTwister(1), mcmc; init_stepsize = 0.125, N = 800, target_acc = 0.25)
            @test isapprox(step, 0.125; atol = 1e-8) # no expansion/bisection needed at all
        end

        @testset "phase 1, 'acceptance too high' branch (expands stepsize upward)" begin
            step = optimize_stepsize(Random.MersenneTwister(2), mcmc; init_stepsize = 0.001, N = 800, target_acc = 0.25)
            @test 0.15 <= accept_ratio(MCMC.sample(Random.MersenneTwister(2), mcmc, 800; stepsize = step)) <= 0.35
        end

        @testset "phase 1, 'acceptance too low' branch (expands stepsize downward)" begin
            step = optimize_stepsize(Random.MersenneTwister(3), mcmc; init_stepsize = 4.0, N = 800, target_acc = 0.25)
            @test 0.15 <= accept_ratio(MCMC.sample(Random.MersenneTwister(3), mcmc, 800; stepsize = step)) <= 0.35
        end

        @testset "phase 2: bisection is actually reached" begin
           
            step = optimize_stepsize(
                Random.MersenneTwister(4),
                mcmc;
                init_stepsize = 0.0625,
                N = 1500,
                target_acc = 0.33,
                tol = 0.05,
            )
            @test 0.0625 < step < 0.125 # strictly inside the bracket -> bisection ran, not just phase 1
        end

        @testset "max_expansions exceeded: 'stayed above target' (expand-up) branch" begin
            let thrown = @test_throws ArgumentError optimize_stepsize(
                    Random.MersenneTwister(5),
                    mcmc;
                    init_stepsize = 0.25,
                    N = 300,
                    target_acc = -0.5,
                    tol = 0.05,
                    max_expansions = 3,
                )
                @test contains(thrown.value.msg, "stayed above")
                @test contains(thrown.value.msg, "doublings")
                @test contains(thrown.value.msg, "max_expansions")
                @test contains(thrown.value.msg, "-0.5")
            end
        end

        @testset "max_expansions exceeded: 'stayed below target' (expand-down) branch" begin
            # target_acc set above the achievable range (>1), symmetric to the case above.
            let thrown = @test_throws ArgumentError optimize_stepsize(
                    Random.MersenneTwister(6),
                    mcmc;
                    init_stepsize = 0.25,
                    N = 300,
                    target_acc = 1.5,
                    tol = 0.05,
                    max_expansions = 3,
                )
                @test contains(thrown.value.msg, "stayed below")
                @test contains(thrown.value.msg, "halvings")
                @test contains(thrown.value.msg, "max_expansions")
                @test contains(thrown.value.msg, "1.5")
            end
        end

        @testset "overall max_iter budget exhausted" begin
            # A reachable target, but with only 1 sample() call allowed in total: the very first
            # (non-converging) phase-1 expansion step must hit the n_evals > max_iter guard.
            let thrown = @test_throws ArgumentError optimize_stepsize(
                    Random.MersenneTwister(7),
                    mcmc;
                    init_stepsize = 4.0,
                    N = 300,
                    target_acc = 0.25,
                    max_iter = 1,
                )
                @test contains(thrown.value.msg, "iteration budget")
                @test contains(thrown.value.msg, "max_iter")
            end
        end
    end

    @testset "Sampler transition-density interface (pCN Hastings-term fix)" begin
        # Regression tests for: pCN's (and Barker's) log_transition_density must reflect the
        # *actual* asymmetric proposal kernel, not the symmetric additive random-walk kernel.
        # Constructed directly at the sampler level (bypassing Emulator/MCMCWrapper) so the
        # underlying transition-density math can be checked against closed-form references.
        rng = Random.MersenneTwister(2026)
        n = 3
        # A genuinely correlated (non-diagonal) covariance, so the tests exercise the
        # whitening/off-diagonal structure, not just a diagonal special case.
        C = [2.0 0.5 0.1; 0.5 1.5 0.3; 0.1 0.3 1.0]
        L = cholesky(Symmetric(C)).L
        prior_dist = MvNormal(zeros(n), Symmetric(C))

        a = randn(rng, n)
        b = randn(rng, n)
        dummy_model = AdvancedMH.DensityModel(θ -> 0.0)

        @testset "RW: transition density is symmetric" begin
            rw = MCMC.RWMetropolisHastings{typeof(L), MCMC.GradFreeProtocol}(L)
            for s in (0.1, 1.0, 2.5)
                @test isapprox(
                    MCMC.log_transition_density(rw, dummy_model, a, b; stepsize = s),
                    MCMC.log_transition_density(rw, dummy_model, b, a; stepsize = s),
                )
            end
        end

        @testset "pCN: Hastings term exactly cancels the prior ratio" begin
            # This is the core bug: reverse - forward must equal logprior(a) - logprior(b)
            # for every stepsize (every ρ), not silently 0.
            pcn = MCMC.pCNMetropolisHastings{typeof(L), MCMC.GradFreeProtocol}(L)
            prev_state = MCMC.MCMCState(a, 0.0, true)
            for s in (0.05, 0.5, 1.5, 3.9)
                hastings = AdvancedMH.logratio_proposal_density(pcn, dummy_model, prev_state, b; stepsize = s)
                @test isapprox(hastings, logpdf(prior_dist, a) - logpdf(prior_dist, b); atol = 1e-8)
            end
        end

        @testset "pCN: flat-likelihood chain always accepts" begin
            # If the "likelihood" is constant, the full posterior log-density is just the prior,
            # and the correct pCN acceptance log-ratio is then EXACTLY 0 for every proposal
            # (loglik cancels trivially, and the Hastings term cancels the prior ratio). Under
            # the pre-fix code (Hastings term silently 0), this would instead fluctuate with the
            # prior ratio and cause spurious rejections.
            pcn = MCMC.pCNMetropolisHastings{typeof(L), MCMC.GradFreeProtocol}(L)
            flat_model = AdvancedMH.DensityModel(θ -> logpdf(prior_dist, θ))
            θ0 = zeros(n)
            state = MCMC.MCMCState(θ0, AdvancedMH.logdensity(flat_model, θ0), true)
            for i in 1:200
                state, _ = AbstractMCMC.step(rng, flat_model, pcn, state; stepsize = 0.6)
                @test state.accepted
            end
        end

        @testset "RW/pCN: propose draws agree with transition_kernel's analytic mean" begin
            # propose() and log_transition_density() are now both derived from the single
            # transition_kernel(...) object (a Distributions.jl MvNormal), so this is really a
            # sanity check that rand(transition_kernel(...)) behaves as documented, rather than a
            # check that two independent formulas happen to agree.
            #
            # n_draws and atol are chosen together for a ~5σ-safe check (not flaky, but tight
            # enough to catch a real scaling bug), via the Monte Carlo mean's norm-error RMS
            # sqrt(tr(Var)/n_draws): RW's Var = stepsize²C gives 5·RMS ≈ 0.053; pCN's Var =
            # (1-ρ²)C gives 5·RMS ≈ 0.067 (both at stepsize=0.5, n_draws=10_000).
            rw = MCMC.RWMetropolisHastings{typeof(L), MCMC.GradFreeProtocol}(L)
            pcn = MCMC.pCNMetropolisHastings{typeof(L), MCMC.GradFreeProtocol}(L)
            state = MCMC.MCMCState(a, 0.0, true)
            n_draws = 10_000
            rw_draws = [AdvancedMH.propose(rng, rw, dummy_model, state; stepsize = 0.5) for _ in 1:n_draws]
            pcn_draws = [AdvancedMH.propose(rng, pcn, dummy_model, state; stepsize = 0.5) for _ in 1:n_draws]
            @test isapprox(mean(rw_draws), a; atol = 0.06)
            ρ = (1 - 0.5 / 4) / (1 + 0.5 / 4)
            @test isapprox(mean(pcn_draws), ρ .* a; atol = 0.07)
        end

        @testset "Barker: transition density matches closed-form gradient formula" begin
            # Target chosen as the same Gaussian, so ∇log π(θ) = -C⁻¹θ is known in closed form,
            # letting us check the whitened, flip-sign transition density against hand-derived
            # algebra rather than trusting the autodiff call alone.
            target_logdensity(θ) = -0.5 * dot(θ, C \ θ)
            barker_model = AdvancedMH.DensityModel(target_logdensity)
            barker = MCMC.BarkerMetropolisHastings{typeof(L), MCMC.ForwardDiffProtocol}(L)
            prev_state = MCMC.MCMCState(a, 0.0, true)
            for s in (0.3, 1.0, 2.0)
                hastings = AdvancedMH.logratio_proposal_density(barker, barker_model, prev_state, b; stepsize = s)

                gw_a = L' * (-(C \ a))
                gw_b = L' * (-(C \ b))
                e = (L \ (b .- a)) ./ s
                sigmoid(x) = 1 / (1 + exp(-x))
                manual = sum(log.(sigmoid.(-gw_b .* e)) .- log.(sigmoid.(gw_a .* e)))
                @test isapprox(hastings, manual; atol = 1e-8)
            end
        end

        @testset "Barker: _barker_rand's marginal matches the 2φ(e)σ(d·e) density" begin
            # Unlike the Hastings-term check above (which only tests reversibility), this checks
            # that _barker_rand's actual empirical distribution matches the closed-form q(e) =
            # 2φ(e)σ(d·e) claimed in its docstring, via a fine-grid quadrature reference — an
            # independent numerical check, not just internal self-consistency.
            d = 1.3
            L1 = reshape([1.0], 1, 1)
            kernel = MCMC.BarkerKernel([0.0], [d], L1, 1.0)

            grid = -12:0.001:12
            dx = step(grid)
            q(e) = 2 * pdf(Normal(), e) * (1 / (1 + exp(-d * e)))
            total_mass = sum(q(e) for e in grid) * dx
            mean_theory = sum(e * q(e) for e in grid) * dx / total_mass
            @test isapprox(total_mass, 1.0; atol = 1e-6)

            # n_draws/atol chosen for a ~5σ-safe check: Var[e] under q (via the same quadrature)
            # is ≈0.76, so SE(n_draws=10_000) ≈ 0.0087 and 5·SE ≈ 0.044.
            n_draws = 10_000
            draws = [MCMC._barker_rand(rng, kernel)[1] for _ in 1:n_draws]
            @test isapprox(mean_theory, sum(draws) / n_draws; atol = 0.05)

            for etest in (-2.0, -0.5, 0.0, 0.7, 3.0)
                @test isapprox(exp(MCMC._barker_logpdf(kernel, [etest])), q(etest); atol = 1e-8)
            end
        end

        @testset "log_transition_density errors loudly for an unimplemented sampler" begin
            # A hypothetical new sampler that forgets to implement log_transition_density must
            # fail loudly (ArgumentError) rather than silently falling back to a symmetric,
            # possibly-wrong correction — this is what the shared interface guards against.
            let thrown = @test_throws ArgumentError MCMC.log_transition_density(_DummyMHSampler(), dummy_model, a, b)
                @test contains(thrown.value.msg, "_DummyMHSampler")
                @test contains(thrown.value.msg, "log_transition_density")
            end
        end
    end

    @testset "accept_ratio errors on a chain missing the :accepted internal" begin
        # A Chains object not produced by this module's sample()/optimize_stepsize() (e.g. hand-built,
        # or with internals filtered out) must fail loudly rather than silently mis-reporting.
        bad_chain = MCMCChains.Chains(rand(5, 3, 1), [:a, :b, :other_internal], (parameters = [:a, :b], internals = [:other_internal]))
        let thrown = @test_throws ArgumentError accept_ratio(bad_chain)
            @test contains(thrown.value.msg, "accept_ratio")
            @test contains(thrown.value.msg, "other_internal")
        end
    end
end
