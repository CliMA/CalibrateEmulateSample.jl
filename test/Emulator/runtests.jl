# Import modules
using Random
using Test
using Statistics
using Distributions
using LinearAlgebra

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers

#build an unknown type
struct MLTester <: Emulators.MachineLearningTool end

function constructor_tests(
    iopairs::PairedDataContainer,
    Σ::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing},
    norm_factors,
    decomposition,
) where {FT <: AbstractFloat}

    input_mean = vec(mean(get_inputs(iopairs), dims = 2)) #column vector
    normalization = sqrt(inv(Symmetric(cov(get_inputs(iopairs), dims = 2))))
    normalization = Emulators.calculate_normalization(get_inputs(iopairs))
    norm_inputs = Emulators.normalize(get_inputs(iopairs), input_mean, normalization)
    # [4.] test emulator preserves the structures
    mlt = MLTester()
    @test_throws ErrorException emulator = Emulator(
        mlt,
        iopairs,
        obs_noise_cov = Σ,
        normalize_inputs = true,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    #build a known type, with defaults
    gp = GaussianProcess(GPJL())
    emulator = Emulator(
        gp,
        iopairs,
        obs_noise_cov = Σ,
        normalize_inputs = false,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )

    # compare SVD/norm/stand with stored emulator version
    test_decomp = emulator.decomposition
    @test test_decomp.V == decomposition.V #(use [:,:] to make it an array)
    @test test_decomp.Vt == decomposition.Vt
    @test test_decomp.S == decomposition.S

    gp = GaussianProcess(GPJL())
    emulator2 = Emulator(
        gp,
        iopairs,
        obs_noise_cov = Σ,
        normalize_inputs = true,
        standardize_outputs = false,
        retained_svd_frac = 1.0,
    )
    train_inputs = get_inputs(emulator2.training_pairs)
    @test train_inputs == norm_inputs
    train_inputs2 = Emulators.normalize(emulator2, get_inputs(iopairs))
    @test train_inputs2 == norm_inputs

    # reverse standardise
    gp = GaussianProcess(GPJL())
    emulator3 = Emulator(
        gp,
        iopairs,
        obs_noise_cov = Σ,
        normalize_inputs = false,
        standardize_outputs = true,
        standardize_outputs_factors = norm_factors,
        retained_svd_frac = 1.0,
    )

    #standardized and decorrelated (sd) data
    s_y, s_y_cov = Emulators.standardize(get_outputs(iopairs), Σ, norm_factors)
    sd_train_outputs = get_outputs(emulator3.training_pairs)
    sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(emulator3.decomposition.S))
    decorrelated_s_y = sqrt_singular_values_inv * emulator3.decomposition.Vt * s_y
    @test decorrelated_s_y == sd_train_outputs
end

@testset "Emulators" begin
    #build some quick data + noise
    m = 50
    d = 6
    p = 10
    x = rand(p, m) #R^3
    y = rand(d, m) #R^6

    # "noise"
    μ = zeros(d)
    Σ = rand(d, d)
    Σ = Σ' * Σ
    noise_samples = rand(MvNormal(μ, Σ), m)
    y += noise_samples

    iopairs = PairedDataContainer(x, y, data_are_columns = true)
    @test get_inputs(iopairs) == x
    @test get_outputs(iopairs) == y

    # [1.] test SVD (2D y version)
    test_SVD = svd(Σ)
    transformed_y, decomposition = Emulators.svd_transform(y, Σ, retained_svd_frac = 1.0)
    @test_throws MethodError Emulators.svd_transform(y, Σ[:, 1], retained_svd_frac = 1.0)
    @test test_SVD.V[:, :] == decomposition.V #(use [:,:] to make it an array)
    @test test_SVD.Vt == decomposition.Vt
    @test test_SVD.S == decomposition.S
    @test size(transformed_y) == size(y)

    # 1D y version
    transformed_y, decomposition2 = Emulators.svd_transform(y[:, 1], Σ, retained_svd_frac = 1.0)
    @test_throws MethodError Emulators.svd_transform(y[:, 1], Σ[:, 1], retained_svd_frac = 1.0)
    @test size(transformed_y) == size(y[:, 1])

    # Reverse SVD
    y_new, y_cov_new =
        Emulators.svd_reverse_transform_mean_cov(reshape(transformed_y, (d, 1)), ones(d, 1), decomposition2)
    @test_throws MethodError Emulators.svd_reverse_transform_mean_cov(transformed_y, ones(d, 1), decomposition2)
    @test_throws MethodError Emulators.svd_reverse_transform_mean_cov(
        reshape(transformed_y, (d, 1)),
        ones(d),
        decomposition2,
    )
    @test size(y_new)[1] == size(y[:, 1])[1]
    @test y_new ≈ y[:, 1]
    @test y_cov_new[1] ≈ Σ


    # Truncation
    transformed_y, trunc_decomposition = Emulators.svd_transform(y[:, 1], Σ, retained_svd_frac = 0.95)
    trunc_size = size(trunc_decomposition.S)[1]
    @test test_SVD.S[1:trunc_size] == trunc_decomposition.S
    @test size(transformed_y)[1] == trunc_size

    # [2.] test Normalization
    # full rank
    input_mean = vec(mean(get_inputs(iopairs), dims = 2)) #column vector
    normalization = sqrt(inv(Symmetric(cov(get_inputs(iopairs), dims = 2))))
    @test all(isapprox.(Emulators.calculate_normalization(get_inputs(iopairs)), normalization, atol = 1e-12))


    norm_inputs = Emulators.normalize(get_inputs(iopairs), input_mean, normalization)
    @test all(isapprox.(norm_inputs, normalization * (get_inputs(iopairs) .- input_mean), atol = 1e-12))
    @test isapprox.(norm(cov(norm_inputs, dims = 2) - I), 0.0, atol = 1e-8)

    # reduced rank
    reduced_inputs = get_inputs(iopairs)[:, 1:(p - 1)]
    input_mean = vec(mean(reduced_inputs, dims = 2)) #column vector
    input_cov = cov(reduced_inputs, dims = 2)
    r = rank(input_cov) # = p-2 
    svd_in = svd(input_cov)
    sqrt_inv_sv = 1 ./ sqrt.(svd_in.S[1:r])
    normalization = Diagonal(sqrt_inv_sv) * svd_in.Vt[1:r, :] # size r x p
    @test all(isapprox.(Emulators.calculate_normalization(reduced_inputs), normalization, atol = 1e-12))

    norm_inputs = Emulators.normalize(reduced_inputs, input_mean, normalization)
    @test size(norm_inputs) == (r, p - 1)
    @test isapprox.(norm(cov(norm_inputs, dims = 2)[1:r, 1:r] - I(r)), 0.0, atol = 1e-12)

    # [3.] test Standardization
    norm_factors = 10.0
    norm_factors = fill(norm_factors, size(y[:, 1])) # must be size of output dim

    s_y, s_y_cov = Emulators.standardize(get_outputs(iopairs), Σ, norm_factors)
    @test s_y == get_outputs(iopairs) ./ norm_factors
    @test s_y_cov == Σ ./ (norm_factors .* norm_factors')

    @testset "Emulators - dense Array Σ" begin
        constructor_tests(iopairs, Σ, norm_factors, decomposition)

        # truncation - only test here, where singular value of Σ differ
        gp = GaussianProcess(GPJL())
        emulator4 = Emulator(
            gp,
            iopairs,
            obs_noise_cov = Σ,
            normalize_inputs = false,
            standardize_outputs = false,
            retained_svd_frac = 0.9,
        )
        trunc_size = size(emulator4.decomposition.S)[1]
        @test test_SVD.S[1:trunc_size] == emulator4.decomposition.S
    end

    @testset "Emulators - Diagonal Σ" begin
        x = rand(3, m) #R^3
        y = rand(d, m) #R^5
        # "noise"
        Σ = Diagonal(rand(d) .+ 1.0)
        noise_samples = rand(MvNormal(zeros(d), Σ), m)
        y += noise_samples
        iopairs = PairedDataContainer(x, y, data_are_columns = true)
        _, decomposition = Emulators.svd_transform(y, Σ, retained_svd_frac = 1.0)

        constructor_tests(iopairs, Σ, norm_factors, decomposition)
    end

    @testset "Emulators - UniformScaling Σ" begin
        x = rand(3, m) #R^3
        y = rand(d, m) #R^5
        # "noise"
        Σ = UniformScaling(1.3)
        noise_samples = rand(MvNormal(zeros(d), Σ), m)
        y += noise_samples
        iopairs = PairedDataContainer(x, y, data_are_columns = true)
        _, decomposition = Emulators.svd_transform(y, Σ, retained_svd_frac = 1.0)

        constructor_tests(iopairs, Σ, norm_factors, decomposition)
    end
end
