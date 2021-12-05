# Import modules
using Random
using Test
using GaussianProcesses
using Statistics 
using Distributions
using ScikitLearn
using LinearAlgebra
@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)

using CalibrateEmulateSample.GaussianProcessEmulator
using CalibrateEmulateSample.DataStorage

@testset "GaussianProcessEmulator" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(41)

    # -------------------------------------------------------------------------
    # Test case 1: 1D input, 1D output
    # -------------------------------------------------------------------------

    # Training data
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(rng, n), 1, n)         # predictors/features: 1 x n
    y = reshape(sin.(x) + 0.05 * randn(rng, n)', 1, n) # predictands/targets: 1 x n

    iopairs = PairedDataContainer(x,y,data_are_columns=true)
    # Construct kernel:
    # Squared exponential kernel (note that hyperparameters are on log scale)
    # with observational noise
    GPkernel = SE(log(1.0), log(1.0))

    # These will be the test inputs at which predictions are made
    new_inputs = reshape([0.0, π/2, π, 3*π/2, 2*π], 1, 5)
    
    # Fit Gaussian Process Regression models. 
    # GaussianProcesses.jl (GPJL) provides two predict functions, predict_y 
    # (which predicts the random variable y(θ)) and predict_y (which predicts 
    # the latent random variable f(θ)). 
    # ScikitLearn's Gaussian process regression (SKLJL) only offers one 
    # predict function, which predicts y.

    # GaussianProcess 1: GPJL, predict_y
    gppackage = GPJL()
    pred_type = YType()

   
    gp1 = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing, 
                   normalized=false, noise_learn=true, 
		   truncate_svd=1.0, standardize=false,
                   prediction_type=pred_type, norm_factor=nothing)
    μ1, σ1² = GaussianProcessEmulator.predict(gp1, new_inputs)
    
    @test gp1.input_output_pairs == iopairs
    @test gp1.input_mean[1] ≈ mean(x) atol=1e-2
    @test gp1.sqrt_inv_input_cov == nothing
    @test gp1.prediction_type == pred_type
    @test vec(μ1) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=0.3
    @test size(μ1)==(1,5)
    @test vec(σ1²) ≈ [0.017, 0.003, 0.004, 0.004, 0.009] atol=1e-2

    # Check if normalization works
    gp1_norm = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, 
                        obs_noise_cov=nothing, normalized=true, 
                        noise_learn=true, truncate_svd=1.0,
			standardize=false, prediction_type=pred_type,
			norm_factor=nothing)
    @test gp1_norm.sqrt_inv_input_cov ≈ [sqrt(1.0 / Statistics.var(x))] atol=1e-4


        
    
    # GaussianProcess 2: GPJL, predict_f
    pred_type = FType()
    gp2 = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing, 
                   normalized=false, noise_learn=true, 
		   truncate_svd=1.0, standardize=false,
                   prediction_type=pred_type, norm_factor=nothing)
    μ2, σ2² = GaussianProcessEmulator.predict(gp2, new_inputs)
    # predict_y and predict_f should give the same mean
    @test μ2 ≈ μ1 atol=1e-6


    # GaussianProcess 3: SKLJL 

    gppackage = SKLJL()
    pred_type = YType()
    var = ConstantKernel(constant_value=1.0)
    se = RBF(1.0)
    GPkernel = var * se

    gp3 = GaussianProcess(iopairs, gppackage; GPkernel=GPkernel, obs_noise_cov=nothing, 
                   normalized=false, noise_learn=true, 
		   truncate_svd=1.0, standardize=false,
                   prediction_type=pred_type, norm_factor=nothing)
   
    μ3, σ3² = GaussianProcessEmulator.predict(gp3, new_inputs)
    @test vec(μ3) ≈ [0.0, 1.0, 0.0, -1.0, 0.0] atol=0.3
    @test vec(σ3²) ≈ [0.016, 0.002, 0.003, 0.004, 0.003] atol=1e-2
  

    # -------------------------------------------------------------------------
    # Test case 2: 2D input, 2D output
    # -------------------------------------------------------------------------

    gppackage = GPJL()
    pred_type = YType()

    # Generate training data
    m = 200                                        # number of training points
    
    p = 2   # input dim 
    d = 2   # output dim
    X = 2.0 * π * rand(rng, p, m)
    
    # G(x1, x2)
    g1x = sin.(X[1, :]) .+ cos.(X[2, :])
    g2x = sin.(X[1, :]) .- cos.(X[2, :])
    gx = zeros(2,m)
    gx[1,:] = g1x
    gx[2,:] = g2x

    # Add noise η
    μ = zeros(d) 
    Σ = 0.08 * [[0.8, 0.2] [0.2, 0.5]] # d x d
    noise_samples = rand(rng, MvNormal(μ, Σ), m)
    
    # y = G(x) + η
    Y = gx .+ noise_samples

    iopairs2 = PairedDataContainer(X,Y,data_are_columns=true)
    @test get_inputs(iopairs2) == X
    @test get_outputs(iopairs2) == Y

    transformed_Y, decomposition = svd_transform(Y, Σ, truncate_svd=1.0)
    @test size(transformed_Y) == size(Y)
    transformed_Y, decomposition = svd_transform(Y[:, 1], Σ, truncate_svd=1.0)
    @test size(transformed_Y) == size(Y[:, 1])

    
    gp4 = GaussianProcess(iopairs2, gppackage, GPkernel=nothing, obs_noise_cov=Σ, 
                          normalized=true, noise_learn=true, 
		          truncate_svd=1.0, standardize=false,
                          prediction_type=pred_type, norm_factor=nothing)

    new_inputs = zeros(2, 4)
    new_inputs[:, 2] = [π/2, π]
    new_inputs[:, 3] = [π, π/2]
    new_inputs[:, 4] = [3*π/2, 2*π]
    
    μ4, σ4² = GaussianProcessEmulator.predict(gp4, new_inputs, transform_to_real=true)

    @test μ4[:, 1] ≈ [1.0, -1.0] atol=0.25
    @test μ4[:, 2] ≈ [0.0, 2.0] atol=0.25
    @test μ4[:, 3] ≈ [0.0, 0.0] atol=0.25
    @test μ4[:, 4] ≈ [0.0, -2.0] atol=0.25
    @test length(σ4²) == size(new_inputs,2)
    @test size(σ4²[1]) == (d, d)

    # Check if standardization works
    norm_factor = 10.0
    norm_factor = fill(norm_factor, size(Y[:,1])) # must be size of output dim
    println(norm_factor)
    gp4_standardized = GaussianProcess(iopairs2, gppackage, GPkernel=nothing, obs_noise_cov=Σ, 
                          normalized=true, noise_learn=true, 
		          truncate_svd=1.0, standardize=true,
                          prediction_type=pred_type, norm_factor=norm_factor)
    cov_est = gp4_standardized.decomposition.V * diagm(gp4_standardized.decomposition.S) *
            gp4_standardized.decomposition.Vt
    @test cov_est ≈ Σ ./ (norm_factor .* norm_factor') atol=1e-2
    @test cov_est ≈ Σ ./ 100.0 atol=1e-2

    # Check if truncation works
    norm_factor = 10.0
    norm_factor = fill(norm_factor, size(Y[:,1])) # must be size of output dim
    println(norm_factor)
    gp4_trunc = GaussianProcess(iopairs2, gppackage, GPkernel=nothing, obs_noise_cov=Σ, 
                          normalized=true, noise_learn=true, 
		          truncate_svd=0.1, standardize=true,
                          prediction_type=pred_type, norm_factor=norm_factor)
    μ4_trunc, σ4_trunc = GaussianProcessEmulator.predict(gp4_trunc, new_inputs, transform_to_real=false)
    μ4_trunc_real, σ4_trunc_real = GaussianProcessEmulator.predict(gp4_trunc, new_inputs, transform_to_real=true)
    @test size(μ4_trunc,1) == 1
    @test size(μ4_trunc_real,1) == 2
    

end
