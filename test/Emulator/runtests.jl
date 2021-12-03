# Import modules
using Random
using Test
using Statistics 
using Distributions
using LinearAlgebra

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataStorage

@testset "Emulators" begin

   
    #build some quick data + noise
    m=50
    d=6
    x = rand(3, m) #R^3
    y = rand(d, m) #R^5
    
    # "noise"
    μ = zeros(d) 
    Σ = rand(d,d) 
    Σ = Σ'*Σ 
    noise_samples = rand(MvNormal(μ, Σ), m)
    y += noise_samples
    
    iopairs = PairedDataContainer(x,y,data_are_columns=true)
    @test get_inputs(iopairs) == x
    @test get_outputs(iopairs) == y

    
    
    # [1.] test SVD (2D y version)
    test_SVD = svd(Σ)
    transformed_y, decomposition = Emulators.svd_transform(y, Σ, truncate_svd=1.0)
    @test_throws AssertionError Emulators.svd_transform(y, Σ[:,1], truncate_svd=1.0)
    @test test_SVD.V[:,:] == decomposition.V #(use [:,:] to make it an array)
    @test test_SVD.Vt == decomposition.Vt
    @test test_SVD.S == decomposition.S
    @test size(test_SVD.S)[1] == decomposition.N
    @test size(transformed_y) == size(y)

    # 1D y version
    transformed_y, decomposition = Emulators.svd_transform(y[:, 1], Σ, truncate_svd=1.0)
    @test_throws AssertionError Emulators.svd_transform(y[:,1], Σ[:,1], truncate_svd=1.0)
    @test size(transformed_y) == size(y[:, 1])
    
    # Reverse SVD
    y_new, y_cov_new = Emulators.svd_reverse_transform_mean_cov(reshape(transformed_y,(d,1)),ones(d,1),decomposition, truncate_svd=1.0)
    @test_throws AssertionError Emulators.svd_reverse_transform_mean_cov(transformed_y,ones(d,1),decomposition, truncate_svd=1.0)
    @test_throws AssertionError Emulators.svd_reverse_transform_mean_cov(reshape(transformed_y,(d,1)),ones(d),decomposition, truncate_svd=1.0)
    @test size(y_new)[1] == size(y[:, 1])[1]
    @test y_new ≈ y[:,1]
    @test y_cov_new[1] ≈ Σ
    
    # Truncation
    transformed_y, decomposition = Emulators.svd_transform(y[:, 1], Σ, truncate_svd=0.95)
    trunc_size = size(decomposition.S)[1]
    @test test_SVD.S[1:trunc_size] == decomposition.S
    @test size(transformed_y)[1] == trunc_size
    
    # [2.] test Normalization
    input_mean = reshape(mean(get_inputs(iopairs), dims=2), :, 1) #column vector
    sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(iopairs), dims=2))))

    norm_inputs = Emulators.normalize(get_inputs(iopairs),input_mean,sqrt_inv_input_cov)
    @test norm_inputs == sqrt_inv_input_cov * (get_inputs(iopairs) .- input_mean)

    # [3.] test Standardization
    norm_factor = 10.0
    norm_factor = fill(norm_factor, size(y[:,1])) # must be size of output dim

    s_y, s_y_cov = Emulators.standardize(get_outputs(iopairs),Σ,norm_factor)
    @test s_y == get_outputs(iopairs)./norm_factor
    @test s_y_cov == Σ ./ (norm_factor.*norm_factor')
    
    
    
#=    # [4.] test emulator preserves the structures
    #build a quick dummy structure
    struct MLTester <: Emulators.MachineLearningTool end
    function build_models!(mlt::MLTester,iopairs) end
    function optimize_hyperparameters!(mlt::MLTester) end
    function predict(mlt::MLTester,new_inputs) return new_inputs end

    mlt = MLTester()
    
    emulator = Emulator(
    mlt,
    iopairs,
    obs_noise_cov=Σ,
    normalize_inputs=true,
    standardize_outputs=false,
    truncate_svd=1.0)
    
    # compare with stored emulator version
    test_decomp = emulator.decomposition
    @test test_decomp.V == decomposition.V #(use [:,:] to make it an array)
    @test test_decomp.Vt == decomposition.Vt
    @test test_decomp.S == decomposition.S
    @test test_decomp.N == decomposition.N

    train_inputs = get_inputs(emulator.training_pairs)
    @assert norm_inputs = train_inputs

    train_inputs2 = Emulators.normalize(emulator,get_inputs(iopairs))
    @assert norm_inputs = train_inputs

  =#  
end
