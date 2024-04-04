# Import modules
using Random
using Test
using GaussianProcesses
using Statistics
using Distributions
using LinearAlgebra
using PyCall
using ScikitLearn
const pykernels = PyNULL()
copy!(pykernels, pyimport_conda("sklearn.gaussian_process.kernels", "scikit-learn=1.1.1"))


using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers

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
    # ScikitLearn's Gaussian process regression (SKLJL) only offers one
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


    em1 = Emulator(
        gp1,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    @test_logs (:warn,) (:warn,) Emulator(
        gp1,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    ) # check that gp1 does not get more models added under second call
    Emulator(
        gp1,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )
    @test length(gp1.models) == 1

    Emulators.optimize_hyperparameters!(em1)

    μ1, σ1² = Emulators.predict(em1, new_inputs)

    @test vec(μ1) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol = 0.3
    @test size(μ1) == (1, 5)
    @test vec(σ1²) ≈ [0.017, 0.003, 0.004, 0.004, 0.009] atol = 1e-2

    # GaussianProcess 2: GPJL, predict_f
    pred_type = FType()

    gp2 = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)

    em2 = Emulator(
        gp2,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    Emulators.optimize_hyperparameters!(em2)

    μ2, σ2² = Emulators.predict(em2, new_inputs)
    # predict_y and predict_f should give the same mean
    @test μ2 ≈ μ1 atol = 1e-6


    # GaussianProcess 3: SKLJL

    gppackage = SKLJL()
    pred_type = YType()
    var = pykernels.ConstantKernel(constant_value = 1.0)
    se = pykernels.RBF(1.0)
    GPkernel = var * se


    gp3 = GaussianProcess(gppackage; kernel = GPkernel, noise_learn = true, prediction_type = pred_type)

    em3 = Emulator(
        gp3,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    @test_logs (:warn,) Emulator(
        gp3,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )
    Emulator(
        gp3,
        iopairs,
        obs_noise_cov = nothing,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )
    @test length(gp3.models) == 1 # check that gp3 does not get more models added under repeated calls

    Emulators.optimize_hyperparameters!(em3)


    #gp3 = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing,
    #               normalized=false, noise_learn=true,
    #	   retained_svd_frac=1.0, standardize=false,
    #               prediction_type=pred_type, norm_factor=nothing)

    μ3, σ3² = Emulators.predict(em3, new_inputs)
    @test vec(μ3) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol = 0.3
    @test vec(σ3²) ≈ [0.016, 0.002, 0.003, 0.004, 0.003] atol = 1e-2


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
    em4_noise_learnt = Emulator(
        gp4_noise_learnt,
        iopairs2,
        obs_noise_cov = Σ,
        normalize_inputs = true,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    gp4 = GaussianProcess(gppackage; kernel = nothing, noise_learn = false, prediction_type = pred_type)

    em4_noise_from_Σ = Emulator(
        gp4,
        iopairs2,
        obs_noise_cov = Σ,
        normalize_inputs = true,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    Emulators.optimize_hyperparameters!(em4_noise_learnt)
    Emulators.optimize_hyperparameters!(em4_noise_from_Σ)

    new_inputs = zeros(2, 4)
    new_inputs[:, 2] = [π / 2, π]
    new_inputs[:, 3] = [π, π / 2]
    new_inputs[:, 4] = [3 * π / 2, 2 * π]

    μ4_noise_learnt, σ4²_noise_learnt = Emulators.predict(em4_noise_learnt, new_inputs, transform_to_real = true)
    tol_mu = 0.25

    @test μ4_noise_learnt[:, 1] ≈ [1.0, -1.0] atol = tol_mu
    @test μ4_noise_learnt[:, 2] ≈ [0.0, 2.0] atol = tol_mu
    @test μ4_noise_learnt[:, 3] ≈ [0.0, 0.0] atol = tol_mu
    @test μ4_noise_learnt[:, 4] ≈ [0.0, -2.0] atol = tol_mu
    @test length(σ4²_noise_learnt) == size(new_inputs, 2)
    @test size(σ4²_noise_learnt[1]) == (d, d)

    μ4_noise_from_Σ, σ4²_noise_from_Σ = Emulators.predict(em4_noise_from_Σ, new_inputs, transform_to_real = true)

    @test μ4_noise_from_Σ[:, 1] ≈ [1.0, -1.0] atol = tol_mu
    @test μ4_noise_from_Σ[:, 2] ≈ [0.0, 2.0] atol = tol_mu
    @test μ4_noise_from_Σ[:, 3] ≈ [0.0, 0.0] atol = tol_mu
    @test μ4_noise_from_Σ[:, 4] ≈ [0.0, -2.0] atol = tol_mu

    # check match between the variances (should be similar at least)
    @test all(isapprox.(σ4²_noise_from_Σ, σ4²_noise_learnt, rtol = 2 * tol_mu))


end
