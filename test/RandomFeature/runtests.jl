using Test
using Random
using RandomFeatures
using LinearAlgebra, Distributions

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.ParameterDistributions
using RandomFeatures

seed = 10101010
rng = Random.MersenneTwister(seed)

@testset "RandomFeatures" begin

    @testset "ScalarRandomFeatureInterface" begin

        input_dim = 2
        n_features = 200
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
        diagonalize_input = false
        #build interface
        n_input_hp = Int(input_dim * (input_dim + 1) / 2) + 2 # see  calculate_n_hyperparameters for details
        n_output_hp = 1

        # prior built from:
        # input:  γ₁*(Cholesky_factor + γ₂ * I )
        # output: γ₃
        # where γᵢ positive
        prior = combine_distributions([
            constrained_gaussian("input_prior_cholesky", 0.0, 1.0, -Inf, Inf, repeats = n_input_hp - 2),
            constrained_gaussian("input_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2),
            constrained_gaussian("output_prior", 1.0, 1.0, 0.0, Inf),
        ])

        optimizer_options = Dict(
            "prior" => prior,
            "n_ensemble" => max(ndims(prior) + 1, 10),
            "n_iteration" => 5,
            "cov_sample_multiplier" => 10.0,
            "inflation" => 1e-4,
            "train_fraction" => 0.8,
            "multithread" => "ensemble",
            "verbose" => false,
        )

        srfi = ScalarRandomFeatureInterface(
            n_features,
            input_dim,
            batch_sizes = batch_sizes,
            rng = rng,
            diagonalize_input = diagonalize_input,
            optimizer_options = optimizer_options,
        )

        @test isa(get_rfms(srfi), Vector{RandomFeatures.Methods.RandomFeatureMethod})
        @test isa(get_fitted_features(srfi), Vector{RandomFeatures.Methods.Fit})
        @test get_batch_sizes(srfi) == batch_sizes
        @test get_n_features(srfi) == n_features
        @test get_input_dim(srfi) == input_dim
        @test get_rng(srfi) == rng
        @test get_diagonalize_input(srfi) == diagonalize_input
        @test get_optimizer_options(srfi) == optimizer_options

        # check defaults 
        srfi2 = ScalarRandomFeatureInterface(n_features, input_dim)
        @test get_batch_sizes(srfi2) === nothing
        @test get_rng(srfi2) == Random.GLOBAL_RNG
        @test get_diagonalize_input(srfi2) == false
        @test get_optimizer_options(srfi2) == optimizer_options # we just set the defaults above

    end

    @testset "VectorRandomFeatureInterface" begin

        rng = Random.MersenneTwister(seed)

        input_dim = 2
        output_dim = 3
        n_features = 200
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
        diagonalize_input = true
        diagonalize_output = false

        n_input_hp = Int(input_dim) + 2
        n_output_hp = Int(output_dim * (output_dim + 1) / 2) + 2

        # prior built from:
        # input:  γ₁*(Cholesky_factor_in + γ₂ * I)
        # output: γ₃*(Cholesky_factor_out + γ₄ * I)
        # where γᵢ positive
        prior = combine_distributions([
            constrained_gaussian("input_prior_diagonal", 1.0, 1.0, 0.0, Inf, repeats = n_input_hp - 2),
            constrained_gaussian("input_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2),
            constrained_gaussian("output_prior_cholesky", 0.0, 1.0, -Inf, Inf, repeats = n_output_hp - 2),
            constrained_gaussian("output_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2),
        ])

        optimizer_options = Dict(
            "prior" => prior,
            "n_ensemble" => max(ndims(prior) + 1, 10),
            "n_iteration" => 5,
            "tikhonov" => 0,
            "cov_sample_multiplier" => 10.0,
            "inflation" => 1e-4,
            "train_fraction" => 0.8,
            "multithread" => "ensemble",
            "verbose" => false,
        )

        #build interfaces
        vrfi = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            rng = rng,
            batch_sizes = batch_sizes,
            diagonalize_input = diagonalize_input,
            diagonalize_output = diagonalize_output,
            optimizer_options = optimizer_options,
        )

        @test isa(get_rfms(vrfi), Vector{RandomFeatures.Methods.RandomFeatureMethod})
        @test isa(get_fitted_features(vrfi), Vector{RandomFeatures.Methods.Fit})
        @test get_batch_sizes(vrfi) == batch_sizes
        @test get_n_features(vrfi) == n_features
        @test get_input_dim(vrfi) == input_dim
        @test get_output_dim(vrfi) == output_dim
        @test get_rng(vrfi) == rng
        @test get_diagonalize_input(vrfi) == diagonalize_input
        @test get_diagonalize_output(vrfi) == diagonalize_output
        @test get_optimizer_options(vrfi) == optimizer_options


        #check defaults
        vrfi2 = VectorRandomFeatureInterface(n_features, input_dim, output_dim, diagonalize_input = diagonalize_input)

        @test get_batch_sizes(vrfi2) === nothing
        @test get_rng(vrfi2) == Random.GLOBAL_RNG
        @test get_diagonalize_input(vrfi2) == true
        @test get_diagonalize_output(vrfi2) == false
        @test get_optimizer_options(vrfi2) == optimizer_options # we just set the defaults above

    end

    @testset "RF within Emulator: 1D -> 1D" begin

        # Training data
        input_dim = 1
        output_dim = 1
        n = 40                                       # number of training points
        x = reshape(2.0 * π * rand(n), 1, n)         # unif(0,2π) predictors/features: 1 x n
        obs_noise_cov = 0.05^2 * I
        y = reshape(sin.(x) + 0.05 * randn(n)', 1, n) # predictands/targets: 1 x n
        iopairs = PairedDataContainer(x, y, data_are_columns = true)

        ntest = 40
        new_inputs = reshape(2.0 * π * rand(ntest), 1, ntest)
        new_outputs = sin.(new_inputs)

        # RF parameters
        n_features = 100
        diagonalize_input = true
        # Scalar RF options to mimic squared-exp ARD kernel
        srfi = ScalarRandomFeatureInterface(n_features, input_dim, diagonalize_input = diagonalize_input, rng = rng)

        # Vector RF options to mimic squared-exp ARD kernel (in 1D)
        vrfi = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            diagonalize_input = diagonalize_input,
            rng = rng,
        )

        # build emulators
        em_srfi = Emulator(srfi, iopairs, obs_noise_cov = obs_noise_cov)
        em_vrfi = Emulator(vrfi, iopairs, obs_noise_cov = obs_noise_cov)

        # just see if it prints something
        @test_logs (:info,) Emulators.optimize_hyperparameters!(em_srfi)
        @test_logs (:info,) Emulators.optimize_hyperparameters!(em_vrfi)

        # predict and test at the new inputs
        tol_μ = 0.1 * ntest
        μs, σs² = Emulators.predict(em_srfi, new_inputs, transform_to_real = true)
        @test size(μs) == (1, ntest)
        @test size(σs²) == (1, ntest)
        @test isapprox.(norm(μs - new_outputs), 0, atol = tol_μ)
        @test all(isapprox.(vec(σs²), 0.05^2 * ones(ntest), atol = 1e-2))

        μv, σv² = Emulators.predict(em_vrfi, new_inputs, transform_to_real = true)
        @test size(μv) == (1, ntest)
        @test size(σv²) == (1, ntest)
        @test isapprox.(norm(μv - new_outputs), 0, atol = tol_μ)
        @test all(isapprox.(vec(σv²), 0.05^2 * ones(ntest), atol = 1e-2))

    end
    @testset "RF within Emulator: 2D -> 2D" begin
        # Generate training data
        n = 100 # number of training points

        input_dim = 2   # input dim
        output_dim = 2   # output dim
        X = 2.0 * π * rand(input_dim, n) # [0,2π]x[0,2π]

        # G(x1, x2)
        g1x = sin.(X[1, :]) .+ cos.(X[2, :])
        g2x = sin.(X[1, :]) .- cos.(X[2, :])
        gx = zeros(2, n)
        gx[1, :] = g1x
        gx[2, :] = g2x

        # Add noise η
        μ = zeros(output_dim)
        Σ = 0.05 * [[0.5, 0.2] [0.2, 0.5]] # d x d
        noise_samples = rand(MvNormal(μ, Σ), n)

        # y = G(x) + η
        Y = gx .+ noise_samples

        iopairs = PairedDataContainer(X, Y, data_are_columns = true)


        # RF parameters
        n_features = 150

        # Test a few options for RF
        # 1) scalar + diag in
        # 2) scalar  
        # 3) vector + diag out
        # 4) vector 

        srfi_diagin = ScalarRandomFeatureInterface(n_features, input_dim, diagonalize_input = true, rng = rng)
        srfi = ScalarRandomFeatureInterface(n_features, input_dim, diagonalize_input = false, rng = rng)

        vrfi_diagout = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            diagonalize_input = false,
            diagonalize_output = true,
            optimizer_options = Dict("train_fraction" => 0.7),
            rng = rng,
        )

        vrfi = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            diagonalize_input = false,
            diagonalize_output = false,
            optimizer_options = Dict("train_fraction" => 0.7),
            rng = rng,
        )

        # build emulators
        # svd: scalar, scalar + diag in, vector + diag out, vector
        # no-svd: vector
        em_srfi_svd_diagin = Emulator(srfi_diagin, iopairs, obs_noise_cov = Σ)
        em_srfi_svd = Emulator(srfi, iopairs, obs_noise_cov = Σ)

        em_vrfi_svd_diagout = Emulator(vrfi_diagout, iopairs, obs_noise_cov = Σ)
        em_vrfi_svd = Emulator(vrfi, iopairs, obs_noise_cov = Σ)

        em_vrfi = Emulator(vrfi, iopairs, obs_noise_cov = Σ, decorrelate = false)

        #TODO truncated SVD option for vector (involves resizing prior)

        ntest = 100
        new_inputs = 2.0 * π * rand(input_dim, ntest) # [0,2π]x[0,2π]

        outx = zeros(2, ntest)
        outx[1, :] = sin.(new_inputs[1, :]) .+ cos.(new_inputs[2, :])
        outx[2, :] = sin.(new_inputs[1, :]) .- cos.(new_inputs[2, :])

        μ_ssd, σ²_ssd = Emulators.predict(em_srfi_svd_diagin, new_inputs, transform_to_real = true)
        μ_ss, σ²_ss = Emulators.predict(em_srfi_svd, new_inputs, transform_to_real = true)
        μ_vsd, σ²_vsd = Emulators.predict(em_vrfi_svd_diagout, new_inputs, transform_to_real = true)
        μ_vs, σ²_vs = Emulators.predict(em_vrfi_svd, new_inputs, transform_to_real = true)
        μ_v, σ²_v = Emulators.predict(em_vrfi, new_inputs)

        tol_μ = 0.1 * ntest * output_dim
        @test isapprox.(norm(μ_ssd - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_ss - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_vsd - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_vs - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_v - outx), 0, atol = 5 * tol_μ) # a more approximate option so likely less good approx

        # An example with the other threading option
        vrfi_tul = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            diagonalize_input = false,
            diagonalize_output = false,
            optimizer_options = Dict("train_fraction" => 0.7, "multithread" => "tullio"),
            rng = rng,
        )
        em_vrfi_svd_tul = Emulator(vrfi_tul, iopairs, obs_noise_cov = Σ)
        μ_vs_tul, σ²_vs_tul = Emulators.predict(em_vrfi_svd_tul, new_inputs, transform_to_real = true)
        @test isapprox.(norm(μ_vs_tul - outx), 0, atol = tol_μ)

    end

end
