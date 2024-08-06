using Test
using Random
using RandomFeatures
using LinearAlgebra, Distributions

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions

seed = 10101010
rng = Random.MersenneTwister(seed)

@testset "RandomFeatures" begin

    @testset "hyperparameter prior interface" begin

        # [1. ] CovarianceStructures
        # Costruction
        eps = 1e-4
        r = 3
        odf = OneDimFactor()
        @test typeof(odf) <: CovarianceStructureType
        df = DiagonalFactor(eps)
        @test typeof(df) <: CovarianceStructureType
        @test get_eps(df) == eps
        @test get_eps(DiagonalFactor()) == Float64(1.0)
        cf = CholeskyFactor(eps)
        @test typeof(cf) <: CovarianceStructureType
        @test get_eps(cf) == eps
        @test get_eps(CholeskyFactor()) == Float64(1.0)
        lrf = LowRankFactor(r, eps)
        @test typeof(lrf) <: CovarianceStructureType
        @test rank(lrf) == r
        @test get_eps(lrf) == eps
        @test get_eps(LowRankFactor(r)) == Float64(1.0)
        hlrf = HierarchicalLowRankFactor(r, eps)
        @test typeof(hlrf) <: CovarianceStructureType
        @test rank(hlrf) == r
        @test get_eps(hlrf) == eps
        @test get_eps(HierarchicalLowRankFactor(r)) == Float64(1.0)

        # calculate_n_hyperparameters
        d = 6
        @test calculate_n_hyperparameters(d, odf) == 0
        @test calculate_n_hyperparameters(d, df) == d
        @test calculate_n_hyperparameters(d, cf) == Int(d * (d + 1) / 2)
        @test calculate_n_hyperparameters(d, lrf) == Int(r + d * r)
        @test calculate_n_hyperparameters(d, hlrf) == Int(r * (r + 1) / 2 + d * r)

        # hyperparameters_from_flat - only check shape
        @test isnothing(hyperparameters_from_flat([1], odf))
        for factor in (df, cf, lrf, hlrf)
            x = ones(calculate_n_hyperparameters(d, factor))
            @test size(hyperparameters_from_flat(x, factor)) == (d, d)
        end

        # build_default_prior - only check size of distribution
        name = "test_name"
        @test isnothing(build_default_prior(name, 0, odf))
        for factor in (df, cf, lrf, hlrf)
            n_hp = calculate_n_hyperparameters(d, factor)
            prior = build_default_prior(name, n_hp, factor)
            @test ndims(prior) == n_hp
        end

        # [2. ] Kernel Structures 
        d = 6
        p = 3
        c_in = lrf
        c_out = cf
        k_sep = SeparableKernel(c_in, c_out)
        @test get_input_cov_structure(k_sep) == c_in
        @test get_output_cov_structure(k_sep) == c_out

        k_nonsep = NonseparableKernel(c_in)
        @test get_cov_structure(k_nonsep) == c_in


        # calculate_n_hyperparameters
        @test calculate_n_hyperparameters(d, p, k_sep) ==
              calculate_n_hyperparameters(d, c_in) + calculate_n_hyperparameters(p, c_out) + 1
        @test calculate_n_hyperparameters(d, p, k_nonsep) == calculate_n_hyperparameters(d * p, c_in) + 1
        k_sep1d = SeparableKernel(odf, odf)
        k_nonsep1d = NonseparableKernel(odf)
        @test calculate_n_hyperparameters(1, 1, k_sep1d) == 2
        @test calculate_n_hyperparameters(1, 1, k_nonsep1d) == 2

        # hyper_parameters_from_flat: not applied with scaling hyperparameter
        x = ones(calculate_n_hyperparameters(d, c_in) + calculate_n_hyperparameters(p, c_out))
        @test size(hyperparameters_from_flat(x, d, p, k_sep)[1]) == (d, d)
        @test size(hyperparameters_from_flat(x, d, p, k_sep)[2]) == (p, p)

        x = ones(calculate_n_hyperparameters(d * p, c_in))
        @test size(hyperparameters_from_flat(x, d, p, k_nonsep)) == (d * p, d * p)

        x = [1] # in the 1D case, return(U,V) = (1x1 matrix, nothing)
        @test size(hyperparameters_from_flat(x, 1, 1, k_sep1d)[1]) == (1, 1)
        @test isnothing(hyperparameters_from_flat(x, 1, 1, k_sep1d)[2])
        @test size(hyperparameters_from_flat(x, 1, 1, k_nonsep1d)) == (1, 1)

        # build_default_prior
        @test ndims(build_default_prior(d, p, k_sep)) ==
              ndims(build_default_prior("input", calculate_n_hyperparameters(d, c_in), c_in)) +
              ndims(build_default_prior("output", calculate_n_hyperparameters(p, c_out), c_out)) +
              1
        @test ndims(build_default_prior(d, p, k_nonsep)) ==
              ndims(build_default_prior("full", calculate_n_hyperparameters(d * p, c_in), c_in)) + 1

        @test ndims(build_default_prior(1, 1, k_sep1d)) == 2
        @test ndims(build_default_prior(1, 1, k_nonsep1d)) == 2


        # test shrinkage utility
        samples = rand(MvNormal(zeros(100), I), 20)
        # normal condition number should be huge around 10^18
        # shrinkage cov will have condition number around 1 and be close to I
        good_cov = shrinkage_cov(samples)
        @test (cond(good_cov) < 1.1) && ((good_cov[1] < 1.2) && (good_cov[1] > 0.8))

        # test NICE utility
        samples = rand(MvNormal(zeros(100), I), 20)
        # normal condition number should be huge around 10^18
        # nice cov will have improved conditioning, does not perform as well at this task as shrinking so has looser bounds
        good_cov = nice_cov(samples)
        @test (cond(good_cov) < 100) && ((good_cov[1] < 1.5) && (good_cov[1] > 0.5))

    end

    @testset "ScalarRandomFeatureInterface" begin

        input_dim = 2
        n_features = 200
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
        #build interface


        # prior built from:
        eps = 1e-8
        kernel_structure = SeparableKernel(CholeskyFactor(eps), OneDimFactor()) # Cholesky factorized input, 1D output
        prior = build_default_prior(input_dim, kernel_structure)

        optimizer_options = Dict(
            "prior" => prior,
            "n_ensemble" => max(ndims(prior) + 1, 10),
            "n_iteration" => 5,
            "scheduler" => DataMisfitController(),
            "n_features_opt" => n_features,
            "cov_sample_multiplier" => 2.0,
            "inflation" => 1e-4,
            "train_fraction" => 0.8,
            "multithread" => "ensemble",
            "accelerator" => DefaultAccelerator(),
            "verbose" => false,
            "cov_correction" => "shrinkage",
        )

        srfi = ScalarRandomFeatureInterface(
            n_features,
            input_dim,
            kernel_structure = kernel_structure,
            batch_sizes = batch_sizes,
            rng = rng,
            optimizer_options = optimizer_options,
        )

        @test isa(get_rfms(srfi), Vector{RandomFeatures.Methods.RandomFeatureMethod})
        @test isa(get_fitted_features(srfi), Vector{RandomFeatures.Methods.Fit})
        @test get_batch_sizes(srfi) == batch_sizes
        @test get_n_features(srfi) == n_features
        @test get_input_dim(srfi) == input_dim
        @test get_rng(srfi) == rng
        @test get_kernel_structure(srfi) == kernel_structure
        # check defaults 
        srfi2 = ScalarRandomFeatureInterface(n_features, input_dim)
        @test get_batch_sizes(srfi2) === nothing
        @test get_rng(srfi2) == Random.GLOBAL_RNG
        @test get_kernel_structure(srfi2) ==
              SeparableKernel(cov_structure_from_string("lowrank", input_dim), OneDimFactor())

        # currently the "scheduler" doesn't always satisfy X() = X(), bug so we need to remove this for now
        for key in keys(optimizer_options)
            if !(key ∈ ["scheduler", "prior", "n_ensemble"])
                @test get_optimizer_options(srfi2)[key] == optimizer_options[key] # we just set the defaults above
            end
        end

    end

    @testset "VectorRandomFeatureInterface" begin

        rng = Random.MersenneTwister(seed)

        input_dim = 2
        output_dim = 3
        n_features = 200
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)

        # prior built from:
        eps = 1e-8
        kernel_structure = SeparableKernel(DiagonalFactor(eps), CholeskyFactor(eps)) # Diagonal input, Cholesky factorized output
        prior = build_default_prior(input_dim, output_dim, kernel_structure)


        optimizer_options = Dict(
            "prior" => prior,
            "n_ensemble" => max(ndims(prior) + 1, 10),
            "n_iteration" => 5,
            "scheduler" => DataMisfitController(),
            "cov_sample_multiplier" => 10.0,
            "n_features_opt" => n_features,
            "tikhonov" => 0,
            "inflation" => 1e-4,
            "train_fraction" => 0.8,
            "multithread" => "ensemble",
            "accelerator" => DefaultAccelerator(),
            "verbose" => false,
            "localization" => EnsembleKalmanProcesses.Localizers.NoLocalization(),
            "cov_correction" => "shrinkage",
        )

        #build interfaces
        vrfi = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = kernel_structure,
            rng = rng,
            batch_sizes = batch_sizes,
            optimizer_options = optimizer_options,
        )

        @test isa(get_rfms(vrfi), Vector{RandomFeatures.Methods.RandomFeatureMethod})
        @test isa(get_fitted_features(vrfi), Vector{RandomFeatures.Methods.Fit})
        @test get_batch_sizes(vrfi) == batch_sizes
        @test get_n_features(vrfi) == n_features
        @test get_input_dim(vrfi) == input_dim
        @test get_output_dim(vrfi) == output_dim
        @test get_kernel_structure(vrfi) == kernel_structure
        @test get_rng(vrfi) == rng
        @test get_optimizer_options(vrfi) == optimizer_options

        #check defaults
        vrfi2 = VectorRandomFeatureInterface(n_features, input_dim, output_dim)

        @test get_batch_sizes(vrfi2) === nothing
        @test get_rng(vrfi2) == Random.GLOBAL_RNG
        @test get_kernel_structure(vrfi2) == SeparableKernel(
            cov_structure_from_string("lowrank", input_dim),
            cov_structure_from_string("lowrank", output_dim),
        )

        for key in keys(optimizer_options)
            if !(key ∈ ["scheduler", "prior", "n_ensemble"])
                @test get_optimizer_options(vrfi2)[key] == optimizer_options[key] # we just set the defaults above
            end
        end

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
        eps = 1e-8
        scalar_ks = SeparableKernel(DiagonalFactor(eps), OneDimFactor()) # Diagonalize input (ARD-type kernel)
        vector_ks = SeparableKernel(DiagonalFactor(eps), CholeskyFactor()) # Diagonalize input (ARD-type kernel)
        # Scalar RF options to mimic squared-exp ARD kernel
        srfi = ScalarRandomFeatureInterface(n_features, input_dim, kernel_structure = scalar_ks, rng = rng)

        # Vector RF options to mimic squared-exp ARD kernel (in 1D)
        vrfi = VectorRandomFeatureInterface(n_features, input_dim, output_dim, kernel_structure = vector_ks, rng = rng)

        # build emulators
        em_srfi = Emulator(srfi, iopairs, obs_noise_cov = obs_noise_cov)
        n_srfi = length(get_rfms(srfi))
        em_vrfi = Emulator(vrfi, iopairs, obs_noise_cov = obs_noise_cov)
        n_vrfi = length(get_rfms(vrfi))

        # test bad case
        optimizer_options = Dict("multithread" => "bad_option")

        srfi_bad = ScalarRandomFeatureInterface(
            n_features,
            input_dim,
            kernel_structure = scalar_ks,
            rng = rng,
            optimizer_options = optimizer_options,
        )
        @test_throws ArgumentError Emulator(srfi_bad, iopairs)

        # test under repeats
        @test_logs (:warn,) Emulator(srfi, iopairs, obs_noise_cov = obs_noise_cov)
        Emulator(srfi, iopairs, obs_noise_cov = obs_noise_cov)
        @test length(get_rfms(srfi)) == n_srfi
        @test_logs (:warn,) Emulator(vrfi, iopairs, obs_noise_cov = obs_noise_cov)
        Emulator(vrfi, iopairs, obs_noise_cov = obs_noise_cov)
        @test length(get_rfms(vrfi)) == n_vrfi


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
        # 5) vector nonseparable
        eps = 1e-8
        r = 1
        scalar_diagin_ks = SeparableKernel(DiagonalFactor(eps), OneDimFactor())
        scalar_ks = SeparableKernel(CholeskyFactor(eps), OneDimFactor())
        vector_diagout_ks = SeparableKernel(CholeskyFactor(eps), DiagonalFactor(eps))
        vector_ks = SeparableKernel(LowRankFactor(r, eps), HierarchicalLowRankFactor(r, eps))
        vector_nonsep_ks = NonseparableKernel(LowRankFactor(r + 1, eps))

        srfi_diagin =
            ScalarRandomFeatureInterface(n_features, input_dim, kernel_structure = scalar_diagin_ks, rng = rng)
        srfi = ScalarRandomFeatureInterface(n_features, input_dim, kernel_structure = scalar_ks, rng = rng)

        vrfi_diagout = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_diagout_ks,
            rng = rng,
        )

        vrfi = VectorRandomFeatureInterface(n_features, input_dim, output_dim, kernel_structure = vector_ks, rng = rng)

        vrfi_nonsep = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_nonsep_ks,
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
        em_vrfi_nonsep = Emulator(vrfi_nonsep, iopairs, obs_noise_cov = Σ, decorrelate = false)

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
        μ_vns, σ²_vns = Emulators.predict(em_vrfi_nonsep, new_inputs)

        tol_μ = 0.1 * ntest * output_dim
        @test isapprox.(norm(μ_ssd - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_ss - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_vsd - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_vs - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_v - outx), 0, atol = 5 * tol_μ) # approximate option so likely less good approx
        @test isapprox.(norm(μ_vns - outx), 0, atol = 5 * tol_μ) # approximate option so likely less good approx

        # An example with the other threading option
        vrfi_tul = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_ks,
            optimizer_options = Dict("train_fraction" => 0.7, "multithread" => "tullio"),
            rng = rng,
        )
        em_vrfi_svd_tul = Emulator(vrfi_tul, iopairs, obs_noise_cov = Σ)
        μ_vs_tul, σ²_vs_tul = Emulators.predict(em_vrfi_svd_tul, new_inputs, transform_to_real = true)
        @test isapprox.(norm(μ_vs_tul - outx), 0, atol = tol_μ)

    end

end
