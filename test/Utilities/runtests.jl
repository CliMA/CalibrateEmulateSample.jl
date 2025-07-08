using Test
using Random
using Statistics
using Distributions
using LinearAlgebra

using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.DataContainers

@testset "Utilities" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(41)

    # test get_training_points
    # first create the EnsembleKalmanProcess
    n_ens = 10
    dim_obs = 3
    dim_par = 2
    initial_ensemble = randn(rng, dim_par, n_ens)#params are cols
    y_obs = randn(rng, dim_obs)
    Γy = Matrix{Float64}(I, dim_obs, dim_obs)
    ekp = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), rng = rng)
    g_ens = randn(rng, dim_obs, n_ens) # data are cols
    EnsembleKalmanProcesses.update_ensemble!(ekp, g_ens)
    training_points = get_training_points(ekp, 1)
    @test get_inputs(training_points) ≈ initial_ensemble
    @test get_outputs(training_points) ≈ g_ens

    #positive definiteness
    mat = reshape(collect(-3:(-3 + 10^2 - 1)), 10, 10)
    tol = 1e12 * eps()
    @test !isposdef(mat)

    pdmat = posdef_correct(mat)
    @test isposdef(pdmat)
    @test minimum(eigvals(pdmat)) >= (1 - 1e-4) * 1e8 * eps() #eigvals is approximate so need a bit of give here  

    pdmat2 = posdef_correct(mat, tol = tol)
    @test isposdef(pdmat2)
    @test minimum(eigvals(pdmat2)) >= (1 - 1e-4) * tol

end


@testset "Data Preprocessing" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(4154)

    # quick build tests and test getters
    zs = zscore_scale()
    mm = minmax_scale()
    qq = quartile_scale()
    QQ = ElementwiseScaler{QuartileScaling, Vector{Int}}([1], [2])
    @test isa(zs, ElementwiseScaler)
    @test get_type(zs) == ZScoreScaling
    @test isa(mm, ElementwiseScaler)
    @test get_type(mm) == MinMaxScaling
    @test isa(qq, ElementwiseScaler)
    @test get_type(qq) == QuartileScaling
    @test get_shift(QQ) == [1]
    @test get_scale(QQ) == [2]

    dd = decorrelate()
    @test get_retain_var(dd) == 1.0
    @test get_decorrelate_with(dd) == "combined"
    dd2 = decorrelate_sample_cov(retain_var = 0.7)
    @test get_retain_var(dd2) == 0.7
    @test get_decorrelate_with(dd2) == "sample_cov"
    dd3 = decorrelate_structure_mat(retain_var = 0.7)
    @test get_retain_var(dd3) == 0.7
    @test get_decorrelate_with(dd3) == "structure_mat"
    DD = Decorrelator([1], [2], [3], 1.0, "test")
    @test get_data_mean(DD) == [1]
    @test get_encoder_mat(DD) == [2]
    @test get_decoder_mat(DD) == [3]


    cc = canonical_correlation()
    @test get_retain_var(cc) == 1.0
    cc2 = canonical_correlation(retain_var = 0.7)
    @test get_retain_var(cc2) == 0.7
    cc3 = CanonicalCorrelation([1], [2], [3], 1.0, "test")
    @test get_data_mean(cc3) == [1]
    @test get_encoder_mat(cc3) == [2]
    @test get_decoder_mat(cc3) == [3]
    @test get_apply_to(cc3) == "test"


    # test equalities
    cc = canonical_correlation()
    cc_copy = canonical_correlation()
    dd = decorrelate()
    dd_copy = decorrelate()
    @test cc == cc_copy
    @test dd == dd_copy

    # get some data as IO pairs for functional tests

    in_dim = 10
    out_dim = 50
    samples = 120

    x = randn(rng, in_dim, in_dim)
    prior_cov = x * x'
    in_data = rand(rng, MvNormal(zeros(in_dim), prior_cov), samples)
    obs_noise_cov = [max(5.0 - abs(i - j), 0.0) for i in 1:out_dim, j in 1:out_dim] # [5 4 3 2 1 0 0 ...] off diagonal
    out_data = rand(rng, MvNormal(-10 * ones(out_dim), obs_noise_cov), samples)

    io_pairs = PairedDataContainer(in_data, out_data)
    test_names = [ # order as in schedules below
        "zscore",
        "quartile",
        "minmax",
        "decorrelate-sample-cov",
        "decorrelate-structure-mat",
        "decorrelate-combined",
        "canonical-correlation",
        "decorrelate-structure-mat-retain-0.95-var",
        "canonical-correlation-0.95-var",
    ]

    # Test encodings-decodings individually
    schedules = [
        (zscore_scale(), "in_and_out"),
        (quartile_scale(), "in_and_out"),
        (minmax_scale(), "in_and_out"),
        (decorrelate_sample_cov(), "in_and_out"),
        (decorrelate_structure_mat(), "in_and_out"),
        (decorrelate(), "in_and_out"), # combined
        (canonical_correlation(), "in_and_out"),
        (decorrelate_structure_mat(retain_var = 0.95), "in_and_out"),
        (canonical_correlation(retain_var = 0.95), "in_and_out"),
    ]

    lossless = [fill(true, 6); fill(false, 4)] # are these lossy approximations? 

    # functional test pipeline
    tol = 1e-12

    for (name, sch, ll_flag) in zip(test_names, schedules, lossless)
        encoder_schedule = create_encoder_schedule(sch)
        (encoded_io_pairs, encoded_prior_cov, encoded_obs_noise_cov) =
            encode_with_schedule!(encoder_schedule, io_pairs, prior_cov, obs_noise_cov)

        (decoded_io_pairs, decoded_prior_cov, decoded_obs_noise_cov) =
            decode_with_schedule(encoder_schedule, encoded_io_pairs, encoded_prior_cov, encoded_obs_noise_cov)
        for (enc_dat, dec_dat, test_dat, enc_covv, dec_covv, test_covv, dim) in zip(
            (get_inputs(encoded_io_pairs), get_outputs(encoded_io_pairs)),
            (get_inputs(decoded_io_pairs), get_outputs(decoded_io_pairs)),
            (get_inputs(io_pairs), get_outputs(io_pairs)),
            (encoded_prior_cov, encoded_obs_noise_cov),
            (decoded_prior_cov, decoded_obs_noise_cov),
            (prior_cov, obs_noise_cov),
            (in_dim, out_dim),
        )
            # univariate "rescaling" tests
            if name == "zscore"
                stat_vec = [[mean(dd), std(dd)] for dd in eachrow(enc_dat)]
                stat_mat = reduce(hcat, stat_vec)
                @test all(isapprox.(stat_mat[1, :], zeros(dim), atol = tol))
                @test all(isapprox.(stat_mat[2, :], ones(dim), atol = tol))

                test_vec = [[mean(dd), std(dd)] for dd in eachrow(test_dat)]
                test_mat = reduce(hcat, test_vec)
                @test isapprox(
                    norm(enc_covv - Diagonal(1 ./ test_mat[2, :]) * test_covv * Diagonal(1 ./ test_mat[2, :])),
                    0.0,
                    atol = tol * dim^2,
                )
            elseif name == "quartile"
                quartiles_vec = [quantile(dd, [0.25, 0.5, 0.75]) for dd in eachrow(enc_dat)]
                quartiles_mat = reduce(hcat, quartiles_vec) # 3 rows: Q1, Q2, and Q3
                @test all(isapprox.(quartiles_mat[2, :], zeros(dim), atol = tol))
                @test all(isapprox.(quartiles_mat[3, :] - quartiles_mat[1, :], ones(dim), atol = tol))
                test_vec = [quantile(dd, [0.25, 0.5, 0.75]) for dd in eachrow(test_dat)]
                test_mat = reduce(hcat, test_vec)
                @test isapprox(
                    norm(
                        enc_covv -
                        Diagonal(1 ./ (test_mat[3, :] - test_mat[1, :])) *
                        test_covv *
                        Diagonal(1 ./ (test_mat[3, :] - test_mat[1, :])),
                    ),
                    0.0,
                    atol = tol * dim^2,
                )
            elseif name == "minmax"
                minmax_vec = [[minimum(dd), maximum(dd)] for dd in eachrow(enc_dat)]
                minmax_mat = reduce(hcat, minmax_vec) # 2 rows: min, max
                @test all(isapprox.(minmax_mat[1, :], zeros(dim), atol = tol))
                @test all(isapprox.(minmax_mat[2, :], ones(dim), atol = tol))
                test_vec = [[minimum(dd), maximum(dd)] for dd in eachrow(test_dat)]
                test_mat = reduce(hcat, test_vec)
                @test isapprox(
                    norm(
                        enc_covv -
                        Diagonal(1 ./ (test_mat[2, :] - test_mat[1, :])) *
                        test_covv *
                        Diagonal(1 ./ (test_mat[2, :] - test_mat[1, :])),
                    ),
                    0.0,
                    atol = tol * dim^2,
                )
            end

            # Multivariate lossless tests
            pop_mean = mean(enc_dat, dims = 2)
            pop_cov = cov(enc_dat, dims = 2)
            dimm = size(pop_cov, 1)
            big_tol = 0.1
            if name == "decorrelate-structure-mat"
                @test all(isapprox.(pop_mean, zeros(dimm), atol = tol))
                @test isapprox(norm(pop_cov - I), 0.0, atol = big_tol * dimm^2) # expect poorly accurate
                @test isapprox(norm(enc_covv - I), 0.0, atol = tol * dimm^2) # expect very accurate

            elseif name == "decorrelate-sample-cov"
                @test all(isapprox.(pop_mean, zeros(dimm), atol = tol))
                @test isapprox(norm(pop_cov - I), 0.0, atol = tol * dimm^2) # expect very accurate
                @test isapprox(norm(enc_covv - I), 0.0, atol = big_tol * dimm^2) # expect poorly accurate, particularly if dimm < dim

            elseif name == "decorrelate-combined"
                @test all(isapprox.(pop_mean, zeros(dimm), atol = tol))
                @test isapprox(norm(pop_cov - I), 0.0, atol = big_tol * dimm^2) # expect poorly accurate
                @test isapprox(norm(enc_covv - I), 0.0, atol = big_tol * dimm^2) # expect poorly accurate
            end

            # Multivariate lossy dim-reduction tests
            if name == "decorrelate-structure-mat-retain-0.95-var"
                svdc = svd(test_covv)
                var_cumsum = cumsum(svdc.S) ./ sum(svdc.S)
                @test var_cumsum[dimm] > 0.95
                @test var_cumsum[dimm - 1] < 0.95
                @test all(isapprox.(pop_mean, zeros(dimm), atol = tol))
                @test isapprox(norm(pop_cov - I), 0.0, atol = big_tol * dimm^2) # expect poorly accurate
                @test isapprox(norm(enc_covv - I), 0.0, atol = tol * dimm^2) # expect very accurate
            end

            # Paired data processor reduction:
            if name == "canonical-correlation"
                @test dimm == min(rank(get_inputs(io_pairs)), rank(get_outputs(io_pairs)))
                @test isapprox(norm(enc_dat * enc_dat' - I), 0.0, atol = tol * dimm^2) # test in or out orthogonality

                # check cross-orthogonality is diagonal (nb this test will be duplicate)
                enc_in = get_inputs(encoded_io_pairs)
                enc_out = get_outputs(encoded_io_pairs)
                @test isapprox(norm(enc_in * enc_out' - Diagonal(diag(enc_in * enc_out'))), 0.0, atol = tol * dimm^2) # test cross - orthogonality

                @test isapprox(norm(enc_out * enc_in' - Diagonal(diag(enc_out * enc_in'))), 0.0, atol = tol * dimm^2) # test cross - orthogonality

                # decoder test is lossless only for smaller dimension
                if dim == min(in_dim, out_dim)
                    @test isapprox(norm(dec_dat - test_dat), 0.0, atol = tol * dim * samples)
                    @test isapprox(norm(dec_covv - test_covv), 0.0, atol = tol * dim^2)
                end

            end

            # test decode approximation of lossless options
            if ll_flag
                # when dimm < dim, loss can occur in some tests
                tol1 = (name == "decorrelate-structure-mat" && dimm < dim) ? big_tol : tol
                tol2 = (name == "decorrelate-sample-cov" && dimm < dim) ? big_tol : tol
                @test isapprox(norm(dec_dat - test_dat), 0.0, atol = tol1 * dim * samples)
                @test isapprox(norm(dec_covv - test_covv), 0.0, atol = tol2 * dim^2)

            end

        end

    end



    # combine a few lossless encoding schedules (lossless requires samples>dims)
    samples = 150 # for full test coverage have samples in_dim < samples < out_dim
    in_data = rand(MvNormal(zeros(in_dim), prior_cov), samples)
    out_data = rand(MvNormal(-10 * ones(out_dim), obs_noise_cov), samples)
    io_pairs = PairedDataContainer(in_data, out_data)

    schedule_builder = [
        (zscore_scale(), "in_and_out"),
        (quartile_scale(), "in"),
        (decorrelate_sample_cov(), "in_and_out"),
        (minmax_scale(), "out"),
        (decorrelate_structure_mat(), "in_and_out"),
        (canonical_correlation(), "in"),
    ]

    # make schedule more parsable

    @test_logs (:warn,) create_encoder_schedule((canonical_correlation(), "bad"))
    @test_logs (:warn,) create_encoder_schedule((zscore_scale(), "bad"))
    func = x -> (get_inputs(x), get_outputs(x))
    bad_encoder_schedule = [(canonical_correlation(), func, "bad")]
    @test_throws ArgumentError encode_with_schedule!(bad_encoder_schedule, io_pairs, prior_cov, obs_noise_cov)
    @test_throws ArgumentError decode_with_schedule(bad_encoder_schedule, io_pairs, prior_cov, obs_noise_cov)
    @test_throws ArgumentError decode_data(canonical_correlation(), func(io_pairs), "bad")


    encoder_schedule = create_encoder_schedule(schedule_builder)

    # encode the data using the schedule
    (encoded_io_pairs, encoded_prior_cov, encoded_obs_noise_cov) =
        encode_with_schedule!(encoder_schedule, io_pairs, prior_cov, obs_noise_cov)

    # decode the data using the schedule
    (decoded_io_pairs, decoded_prior_cov, decoded_obs_noise_cov) =
        decode_with_schedule(encoder_schedule, encoded_io_pairs, encoded_prior_cov, encoded_obs_noise_cov)

    tol = 1e-12
    @test all(isapprox.(get_inputs(io_pairs), get_inputs(decoded_io_pairs), atol = tol))
    @test all(isapprox.(get_outputs(io_pairs), get_outputs(decoded_io_pairs), atol = tol))
    @test isapprox(norm(prior_cov - decoded_prior_cov), 0.0, atol = tol * in_dim^2)
    @test isapprox(norm(obs_noise_cov - decoded_obs_noise_cov), 0.0, atol = tol * out_dim^2)

    # enc/dec just data
    samples = 1 # try one sample
    new_in_data = rand(rng, MvNormal(zeros(in_dim), prior_cov), samples)
    new_out_data = rand(rng, MvNormal(-10 * ones(out_dim), obs_noise_cov), samples)
    id = DataContainer(new_in_data)
    od = DataContainer(new_out_data)

    encoded_id = encode_with_schedule(encoder_schedule, id, "in")
    encoded_od = encode_with_schedule(encoder_schedule, od, "out")
    decoded_id = decode_with_schedule(encoder_schedule, encoded_id, "in")
    decoded_od = decode_with_schedule(encoder_schedule, encoded_od, "out")
    # tests in latent space?
    @test isapprox(norm(get_data(decoded_id) - get_data(id)), 0.0, atol = tol * in_dim * samples)
    @test isapprox(norm(get_data(decoded_od) - get_data(od)), 0.0, atol = tol * out_dim * samples)
    @test_throws ArgumentError encode_with_schedule(encoder_schedule, id, "bad")
    @test_throws ArgumentError decode_with_schedule(encoder_schedule, id, "bad")

    # enc/dec just mats
    encoded_pc = encode_with_schedule(encoder_schedule, prior_cov, "in")
    encoded_oc = encode_with_schedule(encoder_schedule, obs_noise_cov, "out")
    decoded_pc = decode_with_schedule(encoder_schedule, encoded_pc, "in")
    decoded_oc = decode_with_schedule(encoder_schedule, encoded_oc, "out")
    @test isapprox(norm(encoded_pc - encoded_prior_cov), 0.0, atol = tol * in_dim^2)
    @test isapprox(norm(encoded_oc - encoded_obs_noise_cov), 0.0, atol = tol * out_dim^2)
    @test isapprox(norm(decoded_pc - decoded_prior_cov), 0.0, atol = tol * in_dim^2)
    @test isapprox(norm(decoded_oc - decoded_obs_noise_cov), 0.0, atol = tol * out_dim^2)
    @test_throws ArgumentError encode_with_schedule(encoder_schedule, prior_cov, "bad")
    @test_throws ArgumentError decode_with_schedule(encoder_schedule, encoded_pc, "bad")




end
