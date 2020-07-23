# Import modules
using Random
using Test
using GaussianProcesses
using Statistics 
using ScikitLearn
@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)

using CalibrateEmulateSample.GPEmulator

@testset "GPEmulator" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    # -------------------------------------------------------------------------
    # Test case 1: 1D input, 1D output
    # -------------------------------------------------------------------------

    # Training data
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), n, 1)         # predictors/features: n x 1
    y = reshape(sin.(x) + 0.05 * randn(n), n, 1) # predictands/targets: n x 1

    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))

    # These will be the test inputs at which predictions are made
    new_inputs = reshape([0.0, π/2, π, 3*π/2, 2*π], 5, 1)

    # Fit Gaussian Process Regression models. 
    # GaussianProcesses.jl (GPJL) provides two predict functions, predict_y 
    # (which predicts the random variable y(θ)) and predict_y (which predicts 
    # the latent random variable f(θ)). 
    # ScikitLearn's Gaussian process regression (SKLJL) only offers one 
    # predict function, which predicts y.

    # GPObj 1: GPJL, predict_y
    gppackage = GPJL()
    pred_type = YType()

    gpobj1 = GPObj(x, y, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing, 
                   normalized=false, noise_learn=true, 
                   prediction_type=pred_type)
    μ1, σ1² = GPEmulator.predict(gpobj1, new_inputs)

    @test gpobj1.inputs == x
    @test gpobj1.data == y
    @test gpobj1.input_mean[1] ≈ 3.0 atol=1e-2
    @test gpobj1.sqrt_inv_input_cov == nothing
    @test gpobj1.prediction_type == pred_type
    @test vec(μ1) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=0.3
    @assert(size(μ1)==(5,1))
    @test vec(σ1²) ≈ [0.017, 0.003, 0.004, 0.004, 0.009] atol=1e-2

    # Check if normalization works
    gpobj1_norm = GPObj(x, y, gppackage; GPkernel=GPkernel, 
                        obs_noise_cov=nothing, normalized=true, 
                        noise_learn=true, prediction_type=pred_type)
    @test gpobj1_norm.sqrt_inv_input_cov ≈ [sqrt(1.0 / Statistics.var(x))] atol=1e-4


    # GPObj 2: GPJL, predict_f
    pred_type = FType()
    gpobj2 = GPObj(x, y, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing, 
                   normalized=false, noise_learn=true, 
                   prediction_type=pred_type)
    μ2, σ2² = GPEmulator.predict(gpobj2, new_inputs)
    # predict_y and predict_f should give the same mean
    @test μ2 ≈ μ1 atol=1e-6


    # GPObj 3: SKLJL 
    gppackage = SKLJL()
    pred_type = YType()
    var = ConstantKernel(constant_value=1.0)
    se = RBF(1.0)
    GPkernel = var * se

    gpobj3 = GPObj(x, y, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing, 
                   normalized=false, noise_learn=true, 
                   prediction_type=pred_type)
    μ3, σ3² = GPEmulator.predict(gpobj3, new_inputs)
    @test vec(μ3) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=0.3
    @test vec(σ3²) ≈ [0.016, 0.002, 0.003, 0.004, 0.003] atol=1e-2


    # -------------------------------------------------------------------------
    # Test case 2: 2D input, 2D output
    # -------------------------------------------------------------------------

    gppackage = GPJL()
    pred_type = YType()

    # Generate training data
    p = 2   # input dim 
    d = 2   # output dim
    X = 2.0 * π * rand(n, p)

    # G(x1, x2)
    g1x = sin.(X[:, 1]) .+ cos.(X[:, 2])
    g2x = sin.(X[:, 1]) .- cos.(X[:, 2])
    gx = [g1x g2x]

    # Add noise η
    μ = zeros(d) 
    Σ = 0.1 * [[0.8, 0.2] [0.2, 0.5]] # d x d
    noise_samples = rand(MvNormal(μ, Σ), n)' 

    # y = G(x) + η
    Y = gx .+ noise_samples

    transformed_Y, decomposition = svd_transform(Y, Σ)
    @test size(transformed_Y) == size(Y)
    transformed_Y, decomposition = svd_transform(Y[1, :], Σ)
    @test size(transformed_Y) == size(Y[1, :])
    
    gpobj4 = GPObj(X, Y, gppackage, GPkernel=nothing, obs_noise_cov=Σ, 
                   normalized=true, noise_learn=true, 
                   prediction_type=pred_type)

    new_inputs = zeros(4, 2)
    new_inputs[2, :] = [π/2, π]
    new_inputs[3, :] = [π, π/2]
    new_inputs[4, :] = [3*π/2, 2*π]

    μ4, σ4² = GPEmulator.predict(gpobj4, new_inputs, transform_to_real=true)
    @test μ4[1, :] ≈ [1.0, -1.0] atol=0.5
    @test μ4[2, :] ≈ [0.0, 2.0] atol=0.5
    @test μ4[3, :] ≈ [0.0, 0.0] atol=0.5
    @test μ4[4, :] ≈ [0.0, -2.0] atol=0.5
    @test length(σ4²) == size(new_inputs, 1)
    @test size(σ4²[1]) == (d, d)
    
end
