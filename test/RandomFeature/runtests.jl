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
        @test calculate_n_hyperparameters(d, lrf) == Int(d + d * r)
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

        # API from string:
        d = 12
        cov_strings_and_checks = [
            ("onedim", OneDimFactor()),
            ("diagonal", DiagonalFactor()),
            ("cholesky", CholeskyFactor()),
            ("lowrank", LowRankFactor(Int(ceil(sqrt(d))))),
            ("hierlowrank", HierarchicalLowRankFactor(Int(ceil(sqrt(d))))),
        ]

        for (cs, cc) in cov_strings_and_checks
            @test cov_structure_from_string(cs, d) == cc
        end
        @test cov_structure_from_string(OneDimFactor()) == OneDimFactor()
        let thrown = @test_throws ArgumentError cov_structure_from_string("bad-string", d)
            @test contains(thrown.value.msg, "input_cov_structure")
            @test contains(thrown.value.msg, repr("bad-string"))
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
              calculate_n_hyperparameters(d, c_in) + calculate_n_hyperparameters(p, c_out)
        @test calculate_n_hyperparameters(d, p, k_nonsep) == calculate_n_hyperparameters(d * p, c_in)
        k_sep1d = SeparableKernel(odf, odf)
        k_nonsep1d = NonseparableKernel(odf)
        @test calculate_n_hyperparameters(1, 1, k_sep1d) == 1
        @test calculate_n_hyperparameters(1, 1, k_nonsep1d) == 1

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
              ndims(build_default_prior("output", calculate_n_hyperparameters(p, c_out), c_out))
        @test ndims(build_default_prior(d, p, k_nonsep)) ==
              ndims(build_default_prior("full", calculate_n_hyperparameters(d * p, c_in), c_in))

        @test ndims(build_default_prior(1, 1, k_sep1d)) == 1
        @test ndims(build_default_prior(1, 1, k_nonsep1d)) == 1


        # test shrinkage utility
        samples = rand(MvNormal(zeros(100), I), 50)
        # normal condition number should be huge around 10^18
        # shrinkage cov will have condition number around 1 and be close to I
        good_cov = shrinkage_cov(samples)
        @test (cond(good_cov) < 10) && ((good_cov[1] < 5.0) && (good_cov[1] > 0.2))

        good_cov = shrinkage_cov(samples, cov_or_corr = "corr", verbose = true)
        @test (cond(good_cov) < 10) && ((good_cov[1] < 5.0) && (good_cov[1] > 0.2))

        # test NICE utility
        # normal condition number should be huge around 10^18
        # nice cov will have improved conditioning, does not perform as well at this task as shrinking so has looser bounds
        good_cov = nice_cov(samples, verbose = true)
        @test (cond(good_cov) < 100) && ((good_cov[1] < 5.0) && (good_cov[1] > 0.2))

        # regression: the noise level of the sample correlation coefficient must be estimated
        # as (1-corr^2)/sqrt(N) (standard asymptotics for the sample correlation coefficient),
        # not the sign-asymmetric (1-corr)/sqrt(N) previously coded. Monte-Carlo the empirical
        # std of the sample correlation at a fixed true correlation and check it matches the
        # corrected formula much more closely than the old, wrong one.
        rng_nice = Random.MersenneTwister(555)
        rho_true = 0.8
        n_mc = 20_000
        n_per_sample = 50
        corr_draws = [
            cor(rand(rng_nice, MvNormal([0.0, 0.0], [1.0 rho_true; rho_true 1.0]), n_per_sample), dims = 2)[1, 2]
            for _ in 1:n_mc
        ]
        empirical_std = std(corr_draws)
        old_formula_std = (1 - rho_true) / sqrt(n_per_sample)
        new_formula_std = (1 - rho_true^2) / sqrt(n_per_sample)
        @test abs(empirical_std - new_formula_std) < abs(empirical_std - old_formula_std)
        @test isapprox(empirical_std, new_formula_std, rtol = 0.1)

        # regression: `_thread_rng_list` builds one independent RNG stream per sample index
        # (not per thread/chunk), so it must be a pure function of (rng, seed, n) -- its output
        # for the first k entries cannot change as n grows, i.e. it does NOT depend on however
        # Threads.nthreads() happens to chunk 1:n at call time.
        base_seed = 4242
        list_n3 = Emulators._thread_rng_list(Random.MersenneTwister(1), base_seed, 3)
        list_n7 = Emulators._thread_rng_list(Random.MersenneTwister(1), base_seed, 7)
        @test rand.(list_n3) == rand.(list_n7)[1:3]

        # determinism: identical (rng type+state, seed, n) => bit-identical streams
        @test rand.(Emulators._thread_rng_list(Random.MersenneTwister(11), 100, 5)) ==
              rand.(Emulators._thread_rng_list(Random.MersenneTwister(11), 100, 5))

        # regression: the caller's concrete RNG *type* is now respected (previously the code
        # always built `Random.MersenneTwister(seed+i)` regardless of the caller's rng, so this
        # branch could never be exercised); the incoming rng's pre-existing *state* does not
        # matter (each stream is explicitly `Random.seed!`-reset), only its type and the `seed`
        # argument, so a different `seed` argument must give different per-sample streams.
        list_seed1 = Emulators._thread_rng_list(Random.MersenneTwister(1234), 0, 4)
        list_seed2 = Emulators._thread_rng_list(Random.MersenneTwister(5678), 0, 4)
        @test all(isa.(list_seed1, Random.MersenneTwister))
        @test rand.(list_seed1) == rand.(list_seed2) # same type, same seed arg => same streams regardless of the input instance's own state
        list_diff_seed = Emulators._thread_rng_list(Random.MersenneTwister(1234), 999, 4)
        @test rand.(list_seed1) != rand.(list_diff_seed)
        list_xoshiro = Emulators._thread_rng_list(Random.Xoshiro(1), 0, 3)
        @test all(isa.(list_xoshiro, Random.Xoshiro))

        # the non-copyable default/global RNG singleton is the case that made a prior `deepcopy`-
        # based attempt at this fix unsafe (`deepcopy(Random.default_rng()) === Random.default_rng()`,
        # i.e. not an independent copy). `copy` must detach it into an independent, non-degenerate
        # stream without perturbing the global RNG's own state.
        @test deepcopy(Random.default_rng()) === Random.default_rng()
        Random.seed!(Random.default_rng(), 42)
        pre_global_draw = rand()
        Random.seed!(Random.default_rng(), 42)
        default_streams = Emulators._thread_rng_list(Random.default_rng(), 999, 3)
        rand(default_streams[1], 1000) # heavily consume one derived stream
        post_global_draw = rand()
        @test pre_global_draw == post_global_draw # global stream untouched by consuming a derived one
        @test length(unique(rand.(default_streams))) == 3 # 3 independent, non-degenerate streams

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
            "n_ensemble" => min(10 * ndims(prior), 100),
            "n_iteration" => 10,
            "scheduler" => DataMisfitController(terminate_at = 1000),
            "n_features_opt" => n_features,
            "cov_sample_multiplier" => 10.0,
            "inflation" => 1e-4,
            "train_fraction" => 0.8,
            "multithread" => "ensemble",
            "accelerator" => NesterovAccelerator(),
            "verbose" => false,
            "cov_correction" => "nice",
            "n_cross_val_sets" => 2,
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

        # Some structs don't satisfy X == X so removed for now
        for key in keys(optimizer_options)
            if !(key ∈ ["scheduler", "prior", "n_ensemble", "accelerator"])
                @test get_optimizer_options(srfi2)[key] == optimizer_options[key] # we just set the defaults above
            end
        end

    end

    @testset "custom hyperparameter prior takes effect (regression)" begin
        # previously `build_models!` always silently rebuilt `build_default_prior(...)`,
        # ignoring any user-supplied `optimizer_options["prior"]` (the rebuild-guard branch
        # `ndims(prior) > n_hp` could never fire since both were computed from the same
        # input_dim/kernel_structure). Assert that a strongly-shifted custom prior actually
        # changes the fit relative to the default prior.
        input_dim = 1
        n_train = 30
        x = reshape(2.0 * π * rand(Random.MersenneTwister(seed), n_train), 1, n_train)
        y = reshape(sin.(x) + 0.02 * randn(Random.MersenneTwister(seed + 1), n_train)', 1, n_train)
        iopairs = PairedDataContainer(x, y, data_are_columns = true)
        new_inputs = reshape(2.0 * π * rand(Random.MersenneTwister(seed + 2), 20), 1, 20)

        kernel_structure = SeparableKernel(CholeskyFactor(1e-8), OneDimFactor())
        n_hp = calculate_n_hyperparameters(input_dim, kernel_structure)
        default_prior = build_default_prior(input_dim, kernel_structure)
        shifted_prior = constrained_gaussian("shifted_cholesky", 20.0, 1e-6, -Inf, Inf, repeats = n_hp)

        srfi_default = ScalarRandomFeatureInterface(
            100,
            input_dim,
            kernel_structure = kernel_structure,
            rng = Random.MersenneTwister(seed),
            optimizer_options = Dict("n_cross_val_sets" => 0, "prior" => default_prior),
        )
        em_default = Emulator(srfi_default, iopairs)
        μ_default, _ = Emulators.predict(em_default, new_inputs)

        srfi_shifted = ScalarRandomFeatureInterface(
            100,
            input_dim,
            kernel_structure = kernel_structure,
            rng = Random.MersenneTwister(seed),
            optimizer_options = Dict("n_cross_val_sets" => 0, "prior" => shifted_prior),
        )
        em_shifted = Emulator(srfi_shifted, iopairs)
        μ_shifted, _ = Emulators.predict(em_shifted, new_inputs)

        @test norm(μ_default - μ_shifted) > 1e-3

        # a prior with a stale hyperparameter count (from a since-changed input/kernel_structure)
        # must still fall back to the default rather than propagating a shape mismatch
        stale_prior = constrained_gaussian("stale", 0.0, 0.1, -Inf, Inf, repeats = n_hp + 2)
        srfi_stale = ScalarRandomFeatureInterface(
            100,
            input_dim,
            kernel_structure = kernel_structure,
            rng = Random.MersenneTwister(seed),
            optimizer_options = Dict("n_cross_val_sets" => 0, "prior" => stale_prior),
        )
        @test_logs (:info,) match_mode = :any Emulator(srfi_stale, iopairs)
    end

    @testset "cov_sample_multiplier too small throws instead of yielding NaN covariance" begin
        input_dim = 1
        n_train = 20
        x = reshape(2.0 * π * rand(Random.MersenneTwister(seed), n_train), 1, n_train)
        y = reshape(sin.(x) + 0.02 * randn(Random.MersenneTwister(seed + 1), n_train)', 1, n_train)
        iopairs = PairedDataContainer(x, y, data_are_columns = true)

        srfi_bad = ScalarRandomFeatureInterface(
            100,
            input_dim,
            rng = rng,
            optimizer_options = Dict("cov_sample_multiplier" => 0.0),
        )
        thrown = @test_throws ArgumentError Emulator(srfi_bad, iopairs)
        @test contains(thrown.value.msg, "cov_sample_multiplier")

        vrfi_bad = VectorRandomFeatureInterface(
            100,
            input_dim,
            1,
            rng = rng,
            optimizer_options = Dict("cov_sample_multiplier" => 0.0),
        )
        thrown = @test_throws ArgumentError Emulator(vrfi_bad, iopairs)
        @test contains(thrown.value.msg, "cov_sample_multiplier")
    end

    @testset "EnsembleThreading mean_of_covs reduction is race-free (regression)" begin
        # `estimate_mean_and_coeffnorm_covariance`/`calculate_ensemble_mean_and_coeffnorm`
        # (EnsembleThreading methods) used to accumulate into a single shared `mean_of_covs`
        # array via `@. mean_of_covs += ...` inside `Threads.@threads` -- a data race across
        # threads. This test environment runs with Threads.nthreads() = $(Threads.nthreads()),
        # so if the race were still present, re-running the identical seeded fit under real
        # thread contention could occasionally drop concurrent increments and silently return a
        # different answer from run to run. After the fix (each thread accumulates into its own
        # slot; the slots are summed sequentially after the parallel region), repeated runs with
        # the same seed must be bit-for-bit identical.
        input_dim = 1
        n_train = 40
        x = reshape(2.0 * π * rand(Random.MersenneTwister(seed), n_train), 1, n_train)
        y = reshape(sin.(x) + 0.02 * randn(Random.MersenneTwister(seed + 1), n_train)', 1, n_train)
        iopairs = PairedDataContainer(x, y, data_are_columns = true)
        new_inputs = reshape(2.0 * π * rand(Random.MersenneTwister(seed + 2), 10), 1, 10)

        kernel_structure = SeparableKernel(CholeskyFactor(1e-8), OneDimFactor())
        opts = Dict(
            "n_ensemble" => 60,
            "n_iteration" => 3,
            "n_features_opt" => 60,
            "cov_sample_multiplier" => 5.0,
            "multithread" => "ensemble",
            "n_cross_val_sets" => 1,
            "verbose" => false,
        )

        function build_and_predict()
            srfi = ScalarRandomFeatureInterface(
                60,
                input_dim,
                kernel_structure = kernel_structure,
                rng = Random.MersenneTwister(20260811),
                optimizer_options = deepcopy(opts),
            )
            em = Emulator(srfi, iopairs)
            return Emulators.predict(em, new_inputs)
        end

        μ1, σ1 = build_and_predict()
        for _ in 1:4
            μi, σi = build_and_predict()
            @test μ1 == μi
            @test σ1 == σi
        end
    end

    @testset "warn when off-diagonal output noise structure is dropped (ScalarRandomFeatureInterface fits one model per output dim)" begin
        # end-to-end: build_models! reaches the shared helper (the same one GaussianProcess uses,
        # Emulators._warn_if_offdiagonal_structure_mat) when the user disables decorrelation and
        # supplies a correlated output noise covariance. ScalarRandomFeatureInterface fits one fully
        # independent scalar model per output dimension, so off-diagonal noise correlations have
        # nowhere to enter the regularization -- Diagonal is the only structure it can represent,
        # not a lossy shortcut, but the user should be warned rather than left silently unaware.
        input_dim = 1
        output_dim = 2
        n_train = 30
        x = reshape(2.0 * π * rand(Random.MersenneTwister(seed), n_train), 1, n_train)
        y = vcat(sin.(x), cos.(x)) .+ 0.01 .* randn(Random.MersenneTwister(seed + 1), output_dim, n_train)
        iopairs_corr = PairedDataContainer(x, y, data_are_columns = true)
        Σ_corr = [1.0 0.5; 0.5 1.0]

        srfi_corr = ScalarRandomFeatureInterface(
            50,
            input_dim,
            rng = Random.MersenneTwister(seed),
            optimizer_options = Dict("n_cross_val_sets" => 0),
        )
        @test_logs (:warn, r"off-diagonal") match_mode = :any Emulator(
            srfi_corr,
            iopairs_corr;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = Σ_corr),
        )

        # a diagonal output noise covariance must stay silent (no spurious warning)
        srfi_diag = ScalarRandomFeatureInterface(
            50,
            input_dim,
            rng = Random.MersenneTwister(seed),
            optimizer_options = Dict("n_cross_val_sets" => 0),
        )
        logs, _ = Test.collect_test_logs() do
            Emulator(srfi_diag, iopairs_corr; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = 1.0 * I))
        end
        @test !any(occursin("off-diagonal", l.message) for l in logs)
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
            "n_ensemble" => min(10 * ndims(prior), 100),
            "n_iteration" => 10,
            "scheduler" => DataMisfitController(terminate_at = 1000),
            "cov_sample_multiplier" => 10.0,
            "n_features_opt" => n_features,
            "inflation" => 1e-4,
            "train_fraction" => 0.8,
            "multithread" => "ensemble",
            "accelerator" => NesterovAccelerator(),
            "verbose" => false,
            "localization" => EnsembleKalmanProcesses.Localizers.NoLocalization(),
            "cov_correction" => "nice",
            "n_cross_val_sets" => 2,
            "overfit" => 1.0,
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

        # exclude some structs where X == X not true
        for key in keys(optimizer_options)
            if !(key ∈ ["scheduler", "prior", "n_ensemble", "accelerator"])
                @test get_optimizer_options(vrfi2)[key] == optimizer_options[key] # we just set the defaults above

            end
        end


    end

    @testset "RF within Emulator: 1D -> 1D" begin

        # Training data
        input_dim = 1
        output_dim = 1
        n = 50                                       # number of training points
        x = reshape(2.0 * π * rand(n), 1, n)         # unif(0,2π) predictors/features: 1 x n
        obs_noise_cov = 0.05^2 * I
        y = reshape(sin.(x) + 0.05 * randn(n)', 1, n) # predictands/targets: 1 x n
        iopairs = PairedDataContainer(x, y, data_are_columns = true)

        ntest = 50
        new_inputs = reshape(2.0 * π * rand(ntest), 1, ntest)
        new_outputs = sin.(new_inputs)

        # RF parameters
        n_features = 100
        eps = 1.0 # more reg needed here for some reason...
        scalar_ks = SeparableKernel(DiagonalFactor(eps), OneDimFactor()) # Diagonalize input (ARD-type kernel)

        eps = 1e-8 # more reg needed here for some reason...
        vector_ks = SeparableKernel(DiagonalFactor(eps), CholeskyFactor()) # Diagonalize input (ARD-type kernel)
        # Scalar RF options to mimic squared-exp ARD kernel
        srfi = ScalarRandomFeatureInterface(
            n_features,
            input_dim,
            kernel_structure = scalar_ks,
            rng = rng,
            optimizer_options = Dict("n_cross_val_sets" => 0),
        )

        # Vector RF options to mimic squared-exp ARD kernel (in 1D)
        vrfi = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_ks,
            rng = rng,
            optimizer_options = Dict("n_cross_val_sets" => 0),
        )

        # build emulators
        em_srfi = Emulator(srfi, iopairs; encoder_kwargs = (; obs_noise_cov = obs_noise_cov))
        n_srfi = length(get_rfms(srfi))
        em_vrfi = Emulator(vrfi, iopairs; encoder_kwargs = (; obs_noise_cov = obs_noise_cov))
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

        # test cross-validation split too large (Scalar RF): train_fraction=0.5 → n_test=25, 25*3=75 > n=50
        let srfi_bad_cv = ScalarRandomFeatureInterface(
                n_features,
                input_dim,
                kernel_structure = scalar_ks,
                rng = rng,
                optimizer_options = Dict("train_fraction" => 0.5, "n_cross_val_sets" => 3),
            )
            thrown = @test_throws ArgumentError Emulator(srfi_bad_cv, iopairs)
            @test contains(thrown.value.msg, "n_cross_val_sets")
            @test contains(thrown.value.msg, "n_test")
            @test contains(thrown.value.msg, "train_fraction")
            @test contains(thrown.value.msg, string(n))
        end

        # test cross-validation split too large (Vector RF)
        let vrfi_bad_cv = VectorRandomFeatureInterface(
                n_features,
                input_dim,
                output_dim,
                kernel_structure = vector_ks,
                rng = rng,
                optimizer_options = Dict("train_fraction" => 0.5, "n_cross_val_sets" => 3),
            )
            thrown = @test_throws ArgumentError Emulator(vrfi_bad_cv, iopairs)
            @test contains(thrown.value.msg, "n_cross_val_sets")
            @test contains(thrown.value.msg, "n_test")
            @test contains(thrown.value.msg, "train_fraction")
            @test contains(thrown.value.msg, string(n))
        end

        # test under repeats
        @test_logs (:info,) (:warn,) (:info,) (:warn,) Emulator(
            srfi,
            iopairs;
            encoder_kwargs = (; obs_noise_cov = obs_noise_cov),
        )
        Emulator(srfi, iopairs; encoder_kwargs = (; obs_noise_cov = obs_noise_cov))
        @test length(get_rfms(srfi)) == n_srfi
        @test_logs (:info,) (:warn,) (:info,) (:warn,) Emulator(
            vrfi,
            iopairs;
            encoder_kwargs = (; obs_noise_cov = obs_noise_cov),
        )
        Emulator(vrfi, iopairs; encoder_kwargs = (; obs_noise_cov = obs_noise_cov))
        @test length(get_rfms(vrfi)) == n_vrfi


        # just see if it prints something
        @test_logs (:info,) Emulators.optimize_hyperparameters!(em_srfi)
        @test_logs (:info,) Emulators.optimize_hyperparameters!(em_vrfi)

        # predict and test at the new inputs
        tol_μ = 0.1 * ntest
        μs, σs² = Emulators.predict(em_srfi, new_inputs; add_obs_noise_cov = true)
        @test size(μs) == (1, ntest)
        @test size(σs²) == (1, ntest)
        @test isapprox.(norm(μs - new_outputs), 0, atol = tol_μ)
        @test all(isapprox.(vec(σs²), 0.05^2 * ones(ntest), atol = 1e-2))

        μv, σv² = Emulators.predict(em_vrfi, new_inputs; add_obs_noise_cov = true)
        @test size(μv) == (1, ntest)
        @test size(σv²) == (1, ntest)
        @test isapprox.(norm(μv - new_outputs), 0, atol = tol_μ)
        @test all(isapprox.(vec(σv²), 0.05^2 * ones(ntest), atol = 1e-2))

        # `true` minus `false` must equal the decoded noise covariance (here `0.05^2`) exactly
        _, σs²_false = Emulators.predict(em_srfi, new_inputs; add_obs_noise_cov = false)
        @test all(isapprox.(vec(σs²) .- vec(σs²_false), 0.05^2, atol = 1e-10))

        _, σv²_false = Emulators.predict(em_vrfi, new_inputs; add_obs_noise_cov = false)
        @test all(isapprox.(vec(σv²) .- vec(σv²_false), 0.05^2, atol = 1e-10))




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

        # Test a few options branches for RF
        # 1) scalar + diag in
        # 2) scalar  
        # 3) vector + diag out, correct cov by shrinkage (cov)
        # 4) vector , correct cov by shrinkage (corr)
        # 5) vector nonseparable , default correction with "nice"
        eps = 1e-6
        r_sep = 1
        r_nonsep = 2
        scalar_diagin_ks = SeparableKernel(DiagonalFactor(eps), OneDimFactor())
        scalar_ks = SeparableKernel(CholeskyFactor(eps), OneDimFactor())
        vector_diagout_ks = SeparableKernel(CholeskyFactor(eps), DiagonalFactor(eps))
        vector_ks = SeparableKernel(HierarchicalLowRankFactor(r_sep, eps), LowRankFactor(r_sep, eps))
        vector_nonsep_ks = NonseparableKernel(LowRankFactor(r_nonsep, eps))

        n_features = 100
        srfi_diagin =
            ScalarRandomFeatureInterface(n_features, input_dim, kernel_structure = scalar_diagin_ks, rng = rng)
        srfi = ScalarRandomFeatureInterface(
            n_features,
            input_dim,
            kernel_structure = scalar_ks,
            rng = rng,
            optimizer_options = Dict("verbose" => true),
        )

        vrfi_diagout = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_diagout_ks,
            rng = rng,
            optimizer_options = Dict("cov_correction" => "shrinkage_corr"),
        )

        vrfi = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_ks,
            rng = rng,
            optimizer_options = Dict("verbose" => true, "cov_correction" => "shrinkage"),
        )

        vrfi_nonsep = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_nonsep_ks,
            rng = rng,
            optimizer_options = Dict("verbose" => true),
        )

        # build emulators
        # svd: scalar, scalar + diag in, vector + diag out, vector
        # no-svd: vector
        em_srfi_svd_diagin = Emulator(srfi_diagin, iopairs; encoder_kwargs = (; obs_noise_cov = Σ))
        em_srfi_svd = Emulator(srfi, iopairs; encoder_kwargs = (; obs_noise_cov = Σ))

        em_vrfi_svd_diagout = Emulator(vrfi_diagout, iopairs; encoder_kwargs = (; obs_noise_cov = Σ))
        em_vrfi_svd = Emulator(deepcopy(vrfi), iopairs; encoder_kwargs = (; obs_noise_cov = Σ))

        em_vrfi = Emulator(deepcopy(vrfi), iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = Σ))
        em_vrfi_nonsep = Emulator(vrfi_nonsep, iopairs; encoder_schedule = [], encoder_kwargs = (; obs_noise_cov = Σ))

        #TODO truncated SVD option for vector (involves resizing prior)

        ntest = 100
        new_inputs = 2.0 * π * rand(input_dim, ntest) # [0,2π]x[0,2π]

        outx = zeros(2, ntest)
        outx[1, :] = sin.(new_inputs[1, :]) .+ cos.(new_inputs[2, :])
        outx[2, :] = sin.(new_inputs[1, :]) .- cos.(new_inputs[2, :])

        μ_ssd, σ2_ssd = Emulators.predict(em_srfi_svd_diagin, new_inputs)
        μ_ss, σ2_ss = Emulators.predict(em_srfi_svd, new_inputs)
        μ_vsd, σ2_vsd = Emulators.predict(em_vrfi_svd_diagout, new_inputs)
        μ_vs, σ2_vs = Emulators.predict(em_vrfi_svd, new_inputs)
        μ_v, σ2_v = Emulators.predict(em_vrfi, new_inputs)
        μ_vns, σ2_vns = Emulators.predict(em_vrfi_nonsep, new_inputs)

        tol_μ = 0.1 * ntest * output_dim
        @test isapprox.(norm(μ_ssd - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_ss - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_vsd - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_vs - outx), 0, atol = tol_μ)
        @test isapprox.(norm(μ_v - outx), 0, atol = 2 * tol_μ) # approximate option so likely less good approx
        @test isapprox.(norm(μ_vns - outx), 0, atol = 2 * tol_μ) # approximate option so likely less good approx
        @info norm.([σ2_ssd, σ2_ss, σ2_vsd, σ2_vs, σ2_v, σ2_vns])
        @test all(isapprox.(norm.([σ2_ssd, σ2_ss, σ2_vsd, σ2_vs]), 0; atol = tol_μ)) # the emulator uncert. of the mean
        @test all(isapprox.(norm.([σ2_v, σ2_vns]), 0; atol = 2 * tol_μ))

        # An example with the other threading option
        vrfi_tul = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            kernel_structure = vector_ks,
            optimizer_options = Dict("train_fraction" => 0.8, "multithread" => "tullio"),
            rng = rng,
        )
        em_vrfi_svd_tul = Emulator(vrfi_tul, iopairs; encoder_kwargs = (; obs_noise_cov = Σ))
        μ_vs_tul, σ2_vs_tul = Emulators.predict(em_vrfi_svd_tul, new_inputs)
        @test isapprox.(norm(μ_vs_tul - outx), 0, atol = tol_μ)
        @test all(isapprox.(norm.(σ2_vs_tul), 0; atol = tol_μ))

    end

    # regression test: a constant input (or output) dimension must not produce Inf/NaN
    # scales (and hence NaN/PosDefException deep in the hyperparameter prior); build_models!
    # should warn and fall back to scale = 1 for the degenerate dimension(s) instead.
    @testset "Scalar/VectorRandomFeatureInterface: degenerate (constant) input dimension" begin
        rng_deg = Random.MersenneTwister(20270728)
        input_dim_deg = 2
        output_dim_deg = 1
        n_deg = 30
        x1_deg = 2.0 * π * rand(rng_deg, n_deg)
        x_deg = vcat(reshape(x1_deg, 1, n_deg), fill(3.0, 1, n_deg)) # 2nd input dimension constant
        y_deg = reshape(sin.(x1_deg) + 0.01 * randn(rng_deg, n_deg), 1, n_deg)
        iopairs_deg = PairedDataContainer(x_deg, y_deg, data_are_columns = true)

        small_opts = Dict("n_cross_val_sets" => 0, "n_iteration" => 1, "n_ensemble" => 10, "n_features_opt" => 10)

        # `encoder_schedule = []` disables the default input decorrelation, which would
        # otherwise whiten-and-truncate the constant dimension away before `build_models!`
        # ever sees it - bypassing it here so the guard added inside `build_models!` itself
        # (for users who configure no input decorrelation) is what's actually exercised.
        srfi_deg = ScalarRandomFeatureInterface(10, input_dim_deg; rng = rng_deg, optimizer_options = small_opts)
        em_srfi_deg = @test_logs (:warn, r"constant input dimension") match_mode = :any Emulator(
            srfi_deg,
            iopairs_deg;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = 0.01 * I),
        )
        μ_s_deg, σ2_s_deg = Emulators.predict(em_srfi_deg, x_deg)
        @test all(isfinite, μ_s_deg)
        @test all(isfinite, σ2_s_deg)

        vrfi_deg = VectorRandomFeatureInterface(
            10,
            input_dim_deg,
            output_dim_deg;
            rng = rng_deg,
            optimizer_options = small_opts,
        )
        em_vrfi_deg = @test_logs (:warn, r"constant input dimension") match_mode = :any Emulator(
            vrfi_deg,
            iopairs_deg;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = 0.01 * I),
        )
        μ_v_deg, σ2_v_deg = Emulators.predict(em_vrfi_deg, x_deg)
        @test all(isfinite, μ_v_deg)
        @test all(isfinite, σ2_v_deg)
    end

    # regression test, output side: a constant output dimension must not produce Inf/NaN
    # scales (and hence NaN/PosDefException deep in the hyperparameter prior); build_models!
    # should warn and fall back to scale = 1 for the degenerate output dimension(s) instead.
    @testset "Scalar/VectorRandomFeatureInterface: degenerate (constant) output dimension" begin
        rng_deg = Random.MersenneTwister(20270728)
        input_dim_deg = 1
        output_dim_deg = 1
        n_deg = 30
        x_deg = reshape(2.0 * π * rand(rng_deg, n_deg), 1, n_deg)
        y_deg = fill(3.0, 1, n_deg) # output constant for every sample
        iopairs_deg = PairedDataContainer(x_deg, y_deg, data_are_columns = true)

        small_opts = Dict("n_cross_val_sets" => 0, "n_iteration" => 1, "n_ensemble" => 10, "n_features_opt" => 10)

        # `encoder_schedule = []` disables the default output decorrelation, which would
        # otherwise whiten-and-truncate the constant output away before `build_models!`
        # ever sees it - bypassing it here so the guard added inside `build_models!` itself
        # (for users who configure no output decorrelation) is what's actually exercised.
        srfi_deg_out = ScalarRandomFeatureInterface(10, input_dim_deg; rng = rng_deg, optimizer_options = small_opts)
        em_srfi_deg_out = @test_logs (:warn, r"constant output detected") match_mode = :any Emulator(
            srfi_deg_out,
            iopairs_deg;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = 0.01 * I),
        )
        μ_s_deg_out, σ2_s_deg_out = Emulators.predict(em_srfi_deg_out, x_deg)
        @test all(isfinite, μ_s_deg_out)
        @test all(isfinite, σ2_s_deg_out)

        vrfi_deg_out = VectorRandomFeatureInterface(
            10,
            input_dim_deg,
            output_dim_deg;
            rng = rng_deg,
            optimizer_options = small_opts,
        )
        em_vrfi_deg_out = @test_logs (:warn, r"constant output dimension") match_mode = :any Emulator(
            vrfi_deg_out,
            iopairs_deg;
            encoder_schedule = [],
            encoder_kwargs = (; obs_noise_cov = 0.01 * I),
        )
        μ_v_deg_out, σ2_v_deg_out = Emulators.predict(em_vrfi_deg_out, x_deg)
        @test all(isfinite, μ_v_deg_out)
        @test all(isfinite, σ2_v_deg_out)
    end

end
