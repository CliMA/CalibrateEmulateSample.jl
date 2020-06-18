# Import modules
using Random
using GaussianProcesses
using Test

using CalibrateEmulateSample.GPEmulator

@testset "GPEmulator" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

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
    new_inputs = [0.0, π/2, π, 3*π/2, 2*π]
    μ, σ² = GPEmulator.predict(gpobj, new_inputs)

    @test_throws Exception GPObj(x, y, gppackage; GPkernel=GPkernel, 
                                 normalized=true, prediction_type=pred_type)
    @test gpobj.inputs == x
    @test gpobj.data == y
    @test gpobj.input_mean[1] ≈ 3.524 atol=1e-2
    @test gpobj.sqrt_inv_input_cov == nothing
    @test gpobj.prediction_type == pred_type
    @test μ ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=1.0
    @test σ² ≈ [0.859, 0.591, 0.686, 0.645, 0.647] atol=1e-2

end
