# Import modules
using Random
using Test
using GaussianProcesses
using ScikitLearn
@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)

using CalibrateEmulateSample.GPEmulator

@testset "GPEmulator" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    # Training data
    n = 10                            # number of training points
    x = 2.0 * π * rand(n)             # predictors/features
    y = sin.(x) + 0.05 * randn(n)     # predictands/targets


    # observational noise 
    obs_noise = 0.1

    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    se = SE(log(1.0), log(1.0))
    white = Noise(log(sqrt(obs_noise)))
    GPkernel = se + white

    # These will be the test inputs at which predictions are made
    new_inputs = [0.0, π/2, π, 3*π/2, 2*π]

    # Fit Gaussian Process Regression models. 
    # GaussianProcesses.jl (GPJL) provides two predict functions, predict_y 
    # (which predicts the observation random variable y(θ)) and predict_y 
    # (which predicts the latent random variable f(θ)). 
    # ScikitLearn's Gaussian process regression (SKLJL) only offers one 
    # predict function, which predicts y.

    # GPObj 1: GPJL, predict_y
    gppackage = GPJL()
    pred_type = YType()

    gpobj1 = GPObj(x, y, gppackage; GPkernel=GPkernel, normalized=false, 
                   prediction_type=pred_type)
    μ1, σ1² = GPEmulator.predict(gpobj1, new_inputs)
    @test_throws Exception GPObj(x, y, gppackage; GPkernel=GPkernel, 
                                 normalized=true, prediction_type=pred_type)
    @test gpobj1.inputs == x
    @test gpobj1.data == y
    @test gpobj1.input_mean[1] ≈ 3.524 atol=1e-2
    @test gpobj1.sqrt_inv_input_cov == nothing
    @test gpobj1.prediction_type == pred_type
    @test μ1 ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=1.0
    @test σ1² ≈ [0.153, 0.003, 0.006, 0.004, 0.008] atol=1e-2

    # GPObj 2: GPJL, predict_f
    pred_type = FType()

    gpobj2 = GPObj(x, y, gppackage; GPkernel=GPkernel, normalized=false, 
                   prediction_type=pred_type)
    μ2, σ2² = GPEmulator.predict(gpobj2, new_inputs)
    # predict_y and predict_f should give the same mean
    @test μ2 ≈ μ1 atol = 1e-6
    # predict_f should give a smaller variance than predict_y (f is a random
    # random variable with a variance defined by the SE kernel, and y is a 
    # random variable that comes from f, but that is corrupted by observational
    # noise)
    @test all(σ2² < σ1²)

    # GPObj 3: SLKJL 
    gppackage = SKLJL()
    var = ConstantKernel(constant_value=1.0)
    white = WhiteKernel(1.0)
    se = RBF(1.0)
    GPkernel = var * se + white

    gpobj3 = GPObj(reshape(x, (size(x)...,1)), y, gppackage; 
                           GPkernel=GPkernel, normalized=false)
    μ3, σ3² = GPEmulator.predict(gpobj3, 
                                 reshape(new_inputs, size(new_inputs)..., 1))
    @test μ3 ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=1.0
    @test σ3² ≈ [0.150, 0.003, 0.006, 0.004, 0.008] atol=1e-2

end
