# Import modules
using Test
using Random
using GaussianProcesses
using Statistics
using Distributions
using LinearAlgebra
using PythonCall
const pykernels = PythonCall.pynew()
PythonCall.pycopy!(pykernels, pyimport("sklearn.gaussian_process.kernels"))


using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Utilities

@testset "GaussianProcess" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    # -------------------------------------------------------------------------
    # Test case 1: 1D input, 1D output
    # -------------------------------------------------------------------------

    # Training data
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), 1, n)         # predictors/features: 1 x n
    y = reshape(sin.(x) + 0.05 * randn(n)', 1, n) # predictands/targets: 1 x n

    iopairs = PairedDataContainer(x, y, data_are_columns = true)
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))

    # These will be the test inputs at which predictions are made
    new_inputs = reshape([0.0, π / 2, π, 3 * π / 2, 2 * π], 1, 5)

    # Fit Gaussian Process Regression models.
    # GaussianProcesses.jl (GPJL) provides two predict functions, predict_y
    # (which predicts the random variable y(θ)) and predict_y (which predicts
    # the latent random variable f(θ)).
    # ScikitLearn's Gaussian process regression (SKLPy) only offers one
    # predict function, which predicts y.

    ## GaussianProcess 1: GPJL, predict_y unnormalized
    gppackage = GPJL()
    pred_type = YType()

    gp1 = GaussianProcess(
        gppackage;
        kernel = GPkernel,
        noise_learn = true,
        alg_reg_noise = 1e-4,
        prediction_type = pred_type,
    )

    @test gp1.kernel == GPkernel
    @test gp1.noise_learn == true
    @test gp1.prediction_type == pred_type
    @test gp1.alg_reg_noise == 1e-4

    em1 = Emulator(gp1, iopairs, encoder_schedule = [])

    @test_logs (:warn,) Emulator(gp1, iopairs, encoder_schedule = []) # check that gp1 does not get more models added under second call

    Emulator(gp1, iopairs, encoder_schedule = [])
    @test length(gp1.models) == 1

    Emulators.optimize_hyperparameters!(em1)

    let thrown = @test_throws ArgumentError Emulators.optimize_hyperparameters!(10) # not an mlt
        @test contains(thrown.value.msg, "does not implement the required emulator interface")
    end

    μ1, σ1² = Emulators.predict(em1, new_inputs)

    @test vec(μ1) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol = 0.3
    @test size(μ1) == (1, 5)
    @test vec(σ1²) ≈ [0.017, 0.003, 0.004, 0.004, 0.009] atol = 1e-2

    # GaussianProcess 1b: use GPJL to create an abstractGP dist.
    agp = GaussianProcess(AGPJL(); noise_learn = true, alg_reg_noise = 1e-4, prediction_type = pred_type)
    @test_throws ArgumentError Emulator(agp, iopairs, encoder_schedule = [])

    gp1_opt_params = Emulators.get_params(gp1)[1] # one model only
    gp1_opt_param_names = get_param_names(gp1)[1] # one model only

    kernel_params = Dict(
        "log_rbf_len" => gp1_opt_params[1:(end - 2)],
        "log_std_sqexp" => gp1_opt_params[end - 1],
        "log_std_noise" => gp1_opt_params[end],
    )

    em_agp_from_gp1 = Emulator(agp, iopairs, encoder_schedule = [], kernel_params = kernel_params)
    optimize_hyperparameters!(em_agp_from_gp1)
    # skip rebuild:
    @test_logs (:warn,) Emulator(agp, iopairs, encoder_schedule = [], kernel_params = kernel_params)

    μ1b, σ1b² = Emulators.predict(em_agp_from_gp1, new_inputs)

    # gp1 and agp_from_gp2 should give similar predictions
    tol_small = 1e-12
    @test all(isapprox.(μ1, μ1b, atol = tol_small))
    @test size(μ1) == (1, 5)
    @test all(isapprox.(σ1², σ1b², atol = tol_small))


    # GaussianProcess 2: GPJL, predict_f
    pred_type = FType()

    gp2 = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)

    em2 = Emulator(gp2, iopairs, encoder_schedule = [])

    Emulators.optimize_hyperparameters!(em2)

    μ2, σ2² = Emulators.predict(em2, new_inputs)
    # predict_y and predict_f should give the same mean
    @test μ2 ≈ μ1 atol = 1e-6

    # GaussianProcess 3: SKLPy

    gppackage = SKLPy()
    pred_type = YType()
    var = pykernels.ConstantKernel(constant_value = 1.0)
    se = pykernels.RBF(1.0)
    GPkernel = var * se

    gp3 = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    em3 = Emulator(gp3, iopairs, encoder_schedule = [])
    @test_logs (:warn,) Emulator(gp3, iopairs, encoder_schedule = [])
    Emulator(gp3, iopairs, encoder_schedule = [])
    @test length(gp3.models) == 1 # check that gp3 does not get more models added under repeated calls

    Emulators.optimize_hyperparameters!(em3)

    μ3, σ3² = Emulators.predict(em3, new_inputs)
    @test vec(μ3) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol = 0.3
    @test vec(σ3²) ≈ [0.016, 0.002, 0.003, 0.004, 0.003] atol = 1e-2

    gp = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)
    Γ = 0.05I
    em = Emulator(gp, iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = Γ))
    @test gp.regularization[end] == gp.alg_reg_noise * Γ.λ

    # GaussianProcess 3b: SKLJL (deprecated alias for SKLPy)
    @testset "SKLJL depwarn" begin
        # depwarn is emitted
        @test_logs (:warn, r"`SKLJL` is deprecated, use `SKLPy` instead") match_mode = :any GaussianProcess(
            SKLJL();
            kernel = GPkernel,
            noise_learn = true,
            prediction_type = YType(),
        )
        # result is backed by SKLPy and kwargs are forwarded correctly
        gp_skljl = GaussianProcess(SKLJL(); kernel = GPkernel, noise_learn = true, prediction_type = YType())
        @test gp_skljl isa GaussianProcess{SKLPy}
        @test gp_skljl.kernel === GPkernel
        @test gp_skljl.noise_learn == true
    end

    # -------------------------------------------------------------------------
    # Test case 2: 2D input, 2D output
    # -------------------------------------------------------------------------

    gppackage = GPJL()
    pred_type = YType()

    # Generate training data
    m = 100 # number of training points

    p = 2   # input dim
    d = 2   # output dim
    X = 2.0 * π * rand(p, m)

    # G(x1, x2)
    g1x = sin.(X[1, :]) .+ cos.(X[2, :])
    g2x = sin.(X[1, :]) .- cos.(X[2, :])
    gx = zeros(2, m)
    gx[1, :] = g1x
    gx[2, :] = g2x

    # Add noise η
    μ = zeros(d)
    Σ = 0.05 * [[0.5, 0.2] [0.2, 0.5]] # d x d
    noise_samples = rand(MvNormal(μ, Σ), m)

    # y = G(x) + η
    Y = gx .+ noise_samples

    iopairs2 = PairedDataContainer(X, Y, data_are_columns = true)
    @test get_inputs(iopairs2) == X
    @test get_outputs(iopairs2) == Y


    # with noise learning - e.g. we add a kernel to learn the noise (even though we provide the Sigma to the emulator)
    gp4_noise_learnt = GaussianProcess(gppackage; kernel = nothing, noise_learn = true, prediction_type = pred_type)
    # without noise learning, just use the SVD transform to deal with observational noise
    em4_noise_learnt = Emulator(gp4_noise_learnt, iopairs2; encoder_kwargs = (; obs_noise_cov = Σ))

    gp4 = GaussianProcess(gppackage; kernel = nothing, noise_learn = false, prediction_type = pred_type)

    em4_noise_from_Σ = Emulator(gp4, iopairs2; encoder_kwargs = (; obs_noise_cov = Σ))

    # add some kernel bounds
    n_hparams = length(Emulators.get_params(gp4)[1])
    low = repeat([log(1e-4)], n_hparams) # bounds provided in log space
    high = repeat([log(1e4)], n_hparams)
    kernbounds = (low, high)
    n_hparams2 = length(Emulators.get_params(gp4_noise_learnt)[1])
    low2 = repeat([log(1e-4)], n_hparams2) # bounds provided in log space
    high2 = repeat([log(1e4)], n_hparams2)
    kernbounds2 = (low2, high2)

    Emulators.optimize_hyperparameters!(em4_noise_from_Σ; kernbounds = kernbounds)
    Emulators.optimize_hyperparameters!(em4_noise_learnt; kernbounds = kernbounds2)

    new_inputs = zeros(2, 4)
    new_inputs[:, 2] = [π / 2, π]
    new_inputs[:, 3] = [π, π / 2]
    new_inputs[:, 4] = [3 * π / 2, 2 * π]

    # First is just to test deprecation message!
    _, _ = Emulators.predict(em4_noise_learnt, new_inputs; transform_to_real = true)
    _, _ = Emulators.predict(em4_noise_learnt, new_inputs; transform_to_real = false)

    # continue with example
    μ4_noise_learnt, σ4²_noise_learnt = Emulators.predict(em4_noise_learnt, new_inputs; add_obs_noise_cov = true)
    tol_mu = 0.25

    @test μ4_noise_learnt[:, 1] ≈ [1.0, -1.0] atol = tol_mu
    @test μ4_noise_learnt[:, 2] ≈ [0.0, 2.0] atol = tol_mu
    @test μ4_noise_learnt[:, 3] ≈ [0.0, 0.0] atol = tol_mu
    @test μ4_noise_learnt[:, 4] ≈ [0.0, -2.0] atol = tol_mu
    @test length(σ4²_noise_learnt) == size(new_inputs, 2)
    @test size(σ4²_noise_learnt[1]) == (d, d)

    μ4_noise_from_Σ, σ4²_noise_from_Σ = Emulators.predict(em4_noise_from_Σ, new_inputs; add_obs_noise_cov = true)

    @test μ4_noise_from_Σ[:, 1] ≈ [1.0, -1.0] atol = tol_mu
    @test μ4_noise_from_Σ[:, 2] ≈ [0.0, 2.0] atol = tol_mu
    @test μ4_noise_from_Σ[:, 3] ≈ [0.0, 0.0] atol = tol_mu
    @test μ4_noise_from_Σ[:, 4] ≈ [0.0, -2.0] atol = tol_mu

    # check match between the variances (should be similar at least)
    @test all(isapprox.(σ4²_noise_from_Σ, σ4²_noise_learnt, rtol = 2 * tol_mu))


    # GaussianProcess 4b: use GPJL to create an abstractGP dist.
    agp4 = GaussianProcess(AGPJL(); noise_learn = true, prediction_type = pred_type)

    gp4_opt_params = Emulators.get_params(gp4_noise_learnt)
    gp4_opt_param_names = get_param_names(gp4_noise_learnt)

    kernel_params = [
        Dict(
            "log_rbf_len" => model_params[1:(end - 2)],
            "log_std_sqexp" => model_params[end - 1],
            "log_std_noise" => model_params[end],
        ) for model_params in gp4_opt_params
    ]

    em_agp_from_gp4 = Emulator(agp4, iopairs2; encoder_kwargs = (; obs_noise_cov = Σ), kernel_params = kernel_params)

    μ4b, σ4b² = Emulators.predict(em_agp_from_gp4, new_inputs; add_obs_noise_cov = true)

    # gp1 and agp_from_gp2 should give similar predictions
    tol_small = 1e-12
    @test all(isapprox.(μ4b, μ4_noise_learnt, atol = tol_small))
    @test all(isapprox.(σ4b², σ4²_noise_learnt, atol = tol_small))

    @testset "M9: add_obs_noise_cov centralization (regression for M1, all three GP backends)" begin
        # `true` minus `false` must equal the decoded noise covariance Σ exactly, for any backend
        σ4²_false_learnt =
            [Matrix(s) for s in Emulators.predict(em4_noise_learnt, new_inputs; add_obs_noise_cov = false)[2]]
        σ4²_true_learnt = [Matrix(s) for s in σ4²_noise_learnt]
        @test all(isapprox.(σ4²_true_learnt .- σ4²_false_learnt, [Σ for _ in σ4²_true_learnt], atol = 1e-10))

        σ4b²_false = [Matrix(s) for s in Emulators.predict(em_agp_from_gp4, new_inputs; add_obs_noise_cov = false)[2]]
        σ4b²_true = [Matrix(s) for s in σ4b²]
        @test all(isapprox.(σ4b²_true .- σ4b²_false, [Σ for _ in σ4b²_true], atol = 1e-10))

        # M1 regression: with `noise_learn = true`, `false` must be a purely latent variance, much
        # smaller than the known noise variance, for GPJL, SKLPy, and AGPJL alike.
        n_dense = 200
        x_dense = reshape(collect(range(0, 2π, length = n_dense)), 1, n_dense)
        σ_noise = 0.05
        y_dense = reshape(sin.(x_dense[1, :]) .+ σ_noise .* randn(n_dense), 1, n_dense)
        iopairs_dense = PairedDataContainer(x_dense, y_dense, data_are_columns = true)
        Σ_dense = σ_noise^2 * ones(1, 1)
        x_interior = reshape([π], 1, 1) # densely sampled interior point: low epistemic uncertainty

        gp_dense = GaussianProcess(GPJL(); noise_learn = true)
        em_dense_gpjl = Emulator(gp_dense, iopairs_dense; encoder_kwargs = (; obs_noise_cov = Σ_dense))
        Emulators.optimize_hyperparameters!(em_dense_gpjl)
        _, σ2_false_gpjl = Emulators.predict(em_dense_gpjl, x_interior; add_obs_noise_cov = false)
        _, σ2_true_gpjl = Emulators.predict(em_dense_gpjl, x_interior; add_obs_noise_cov = true)
        @test only(σ2_false_gpjl) < 0.3 * σ_noise^2
        @test only(σ2_true_gpjl) - only(σ2_false_gpjl) ≈ σ_noise^2 atol = 1e-10

        gp_dense_sklpy = GaussianProcess(SKLPy(); noise_learn = true)
        em_dense_sklpy = Emulator(gp_dense_sklpy, iopairs_dense; encoder_kwargs = (; obs_noise_cov = Σ_dense))
        _, σ2_false_sklpy = Emulators.predict(em_dense_sklpy, x_interior; add_obs_noise_cov = false)
        _, σ2_true_sklpy = Emulators.predict(em_dense_sklpy, x_interior; add_obs_noise_cov = true)
        @test only(σ2_false_sklpy) < 0.3 * σ_noise^2
        @test only(σ2_true_sklpy) - only(σ2_false_sklpy) ≈ σ_noise^2 atol = 1e-10

        gp_dense_params = Emulators.get_params(gp_dense)
        kernel_params_dense = [
            Dict(
                "log_rbf_len" => model_params[1:(end - 2)],
                "log_std_sqexp" => model_params[end - 1],
                "log_std_noise" => model_params[end],
            ) for model_params in gp_dense_params
        ]
        agp_dense = GaussianProcess(AGPJL(); noise_learn = true)
        em_dense_agpjl = Emulator(
            agp_dense,
            iopairs_dense;
            encoder_kwargs = (; obs_noise_cov = Σ_dense),
            kernel_params = kernel_params_dense,
        )
        _, σ2_false_agpjl = Emulators.predict(em_dense_agpjl, x_interior; add_obs_noise_cov = false)
        _, σ2_true_agpjl = Emulators.predict(em_dense_agpjl, x_interior; add_obs_noise_cov = true)
        @test only(σ2_false_agpjl) < 0.3 * σ_noise^2
        @test only(σ2_true_agpjl) - only(σ2_false_agpjl) ≈ σ_noise^2 atol = 1e-10
    end

    @testset "m7: warn when off-diagonal output noise structure is dropped (GP fits one model per encoded dim)" begin
        # direct unit check of the shared helper, across the exact tolerance boundary the fix
        # needs to get right (a naive exact `isdiag` reintroduces false positives on numerically
        # near-diagonal matrices produced by decorrelation)
        @test_logs (:warn, r"off-diagonal") match_mode = :any Emulators._warn_if_offdiagonal_structure_mat(
            [1.0 0.5; 0.5 1.0],
            "GPJL",
        )
        @test_logs Emulators._warn_if_offdiagonal_structure_mat([1.0 0.0; 0.0 1.0], "GPJL") # exactly diagonal: silent
        @test_logs Emulators._warn_if_offdiagonal_structure_mat([1.0 1e-14; 1e-14 1.0], "GPJL") # numerical residual: silent

        # end-to-end: build_models! actually reaches the warning when the user disables
        # decorrelation and supplies a correlated noise covariance, for all three backends
        Σ_corr = [1.0 0.5; 0.5 1.0]

        gp_corr = GaussianProcess(GPJL(); kernel = nothing, noise_learn = false)
        @test_logs (:warn, r"off-diagonal") match_mode = :any Emulator(
            gp_corr,
            iopairs2;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = Σ_corr),
        )

        gp_corr_sklpy =
            GaussianProcess(SKLPy(); kernel = pykernels.ConstantKernel(1.0) * pykernels.RBF(1.0), noise_learn = false)
        @test_logs (:warn, r"off-diagonal") match_mode = :any Emulator(
            gp_corr_sklpy,
            iopairs2;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = Σ_corr),
        )

        agp_corr = GaussianProcess(AGPJL(); noise_learn = false)
        kernel_params_corr =
            [Dict("log_rbf_len" => [0.0, 0.0], "log_std_sqexp" => 0.0, "log_std_noise" => log(1e-6)) for _ in 1:2]
        @test_logs (:warn, r"off-diagonal") match_mode = :any Emulator(
            agp_corr,
            iopairs2;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = Σ_corr),
            kernel_params = kernel_params_corr,
        )
    end

    @testset "h11: predict preserves input eltype (no silent Float64 promotion)" begin
        # GaussianProcesses.jl (GPJL) cannot mix precisions between a fitted model and a query of a
        # different eltype (it errors deep in its own BLAS calls), so only SKLPy's Python-backed
        # predict — which explicitly `pyconvert`s to the query eltype — is exercised end-to-end here.
        # This pins the fix to the `max.(σ2 .- white_var, 1e-12)` promotion (an untyped literal and a
        # Float64 `white_var` silently upcasting an FT=Float32 result) in both GPJL's and SKLPy's
        # `predict` wrappers.
        gp_f32 =
            GaussianProcess(SKLPy(); kernel = pykernels.ConstantKernel(1.0) * pykernels.RBF(1.0), noise_learn = false)
        em_f32 = Emulator(gp_f32, iopairs; encoder_schedule = [])
        new_inputs_f32 = Float32.(reshape([0.0, 1.0, 2.0], 1, 3))
        μ_f32, σ2_f32 = Emulators.predict(gp_f32, new_inputs_f32)
        @test eltype(μ_f32) == Float32
        @test eltype(σ2_f32) == Float32
    end

    @testset "Analytic GP posterior check (test-coverage gap 3): predict matches closed-form k*'(K+σ²I)⁻¹y" begin
        # Independent reference: hand-coded squared-exponential GP regression posterior,
        # k(x,x') = σf² exp(-0.5 (x-x')²/ℓ²), on a tiny fixed dataset, checked against all three
        # backends with FIXED (non-optimized) hyperparameters. This tests the shared data plumbing
        # (regularization scaling, encode/decode with encoder_schedule=[]) against the textbook GP
        # regression equations directly, rather than only cross-checking backends against each other.
        ℓ_a = 1.3
        σf_a = 0.9
        σn2_a = 0.02
        se_kernel(x, xp) = σf_a^2 * exp(-0.5 * (x - xp)^2 / ℓ_a^2)

        x_train_a = [0.1, 0.4, 0.9, 1.5, 2.2]
        y_train_a = [0.2, -0.3, 0.5, 0.1, -0.4]
        x_test_a = [0.3, 1.0, 2.0, 3.0]

        K_a = [se_kernel(xi, xj) for xi in x_train_a, xj in x_train_a]
        Kn_a = K_a + σn2_a * I
        alpha_a = Kn_a \ y_train_a
        Kstar_a = [se_kernel(xi, xj) for xi in x_test_a, xj in x_train_a] # n_test x n_train
        mean_analytic = Kstar_a * alpha_a
        var_analytic = [σf_a^2 - Kstar_a[i, :]' * (Kn_a \ Kstar_a[i, :]) for i in 1:length(x_test_a)]

        iopairs_a = PairedDataContainer(reshape(x_train_a, 1, :), reshape(y_train_a, 1, :), data_are_columns = true)
        Σ_a = σn2_a * ones(1, 1)
        new_inputs_a = reshape(x_test_a, 1, :)

        # GPJL
        gp_a_gpjl = GaussianProcess(GPJL(); kernel = SE(log(ℓ_a), log(σf_a)), noise_learn = false)
        em_a_gpjl = Emulator(gp_a_gpjl, iopairs_a; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = Σ_a))
        μ_a_gpjl, σ2_a_gpjl = Emulators.predict(em_a_gpjl, new_inputs_a; add_obs_noise_cov = false)
        @test vec(μ_a_gpjl) ≈ mean_analytic atol = 1e-8
        @test vec(σ2_a_gpjl) ≈ var_analytic atol = 1e-8

        # SKLPy (bounds fixed so `.fit()` does not re-optimize the given hyperparameters)
        var_kern_a = pykernels.ConstantKernel(constant_value = σf_a^2, constant_value_bounds = "fixed")
        rbf_kern_a = pykernels.RBF(length_scale = ℓ_a, length_scale_bounds = "fixed")
        gp_a_sklpy = GaussianProcess(SKLPy(); kernel = var_kern_a * rbf_kern_a, noise_learn = false)
        em_a_sklpy = Emulator(gp_a_sklpy, iopairs_a; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = Σ_a))
        μ_a_sklpy, σ2_a_sklpy = Emulators.predict(em_a_sklpy, new_inputs_a; add_obs_noise_cov = false)
        @test vec(μ_a_sklpy) ≈ mean_analytic atol = 1e-8
        @test vec(σ2_a_sklpy) ≈ var_analytic atol = 1e-8

        # AGPJL (kernel_params supplied directly; no internal optimization ever occurs)
        agp_a = GaussianProcess(AGPJL(); noise_learn = false)
        kernel_params_a = Dict("log_rbf_len" => [log(ℓ_a)], "log_std_sqexp" => log(σf_a), "log_std_noise" => log(1e-6))
        em_a_agpjl = Emulator(
            agp_a,
            iopairs_a;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = Σ_a),
            kernel_params = kernel_params_a,
        )
        μ_a_agpjl, σ2_a_agpjl = Emulators.predict(em_a_agpjl, new_inputs_a; add_obs_noise_cov = false)
        @test vec(μ_a_agpjl) ≈ mean_analytic atol = 1e-8
        @test vec(σ2_a_agpjl) ≈ var_analytic atol = 1e-8
    end

end
