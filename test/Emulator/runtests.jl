include("MWE.jl")

# Import modules
using Random
using Test
using Statistics
using Distributions
using LinearAlgebra

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Utilities

#build an unknown type
struct MLTester <: Emulators.MachineLearningTool end

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

    io_pairs = PairedDataContainer(x, y, data_are_columns = true)
    @test get_inputs(io_pairs) == x
    @test get_outputs(io_pairs) == y

    # Unknown ML type
    mlt = MLTester()
    @test_throws ErrorException emulator = Emulator(mlt, io_pairs)

    # test getters & defaults
    gp = GaussianProcess(GPJL())
    em = Emulator(gp, io_pairs)
    @test get_machine_learning_tool(em) == gp
    @test get_io_pairs(em) == io_pairs
    default_encoder = (decorrelate_sample_cov(), "in_and_out") # for these inputs this is the default
    enc_sch = create_encoder_schedule(default_encoder)
    enc_io_pairs, enc_I_in, enc_I_out = encode_with_schedule!(enc_sch, io_pairs, 1.0 * I(p), 1.0 * I(d))
    @test get_encoder_schedule(em)[1][1] == enc_sch[1][1] # inputs: proc
    @test get_encoder_schedule(em)[1][3] == enc_sch[1][3] # inputs: apply_to
    @test get_encoder_schedule(em)[2][1] == enc_sch[2][1] # outputs...
    @test get_encoder_schedule(em)[2][3] == enc_sch[2][3]
    @test get_data(get_encoded_io_pairs(em)) == get_data(enc_io_pairs)
    @test get_data(get_encoded_io_pairs(em)) == get_data(enc_io_pairs)

    #NB - encoders all tested in Utilities here just testing some API
    encoded_mat = encode_data(em, x, "in")
    encoded_dc = encode_data(em, DataContainer(x), "in")
    encoded_I = encode_structure_matrix(em, I(d), "out")
    tol = 1e-14
    @test isapprox(norm(encoded_mat - get_data(encoded_dc)), 0, atol = tol * p * m)
    @test isapprox(norm(encoded_mat - get_inputs(enc_io_pairs)), 0, atol = tol * p * m)
    @test isapprox(norm(encoded_I - enc_I_out), 0, atol = tol * d * d)
    decoded_dc = decode_data(em, encoded_dc, "in")
    decoded_mat = decode_data(em, encoded_mat, "in")
    decoded_I = decode_structure_matrix(em, encoded_I, "out")
    @test isapprox(norm(get_data(decoded_dc) - decoded_mat), 0, atol = tol * p * m)
    @test isapprox(norm(decoded_mat - x), 0, atol = tol * p * m)
    @test isapprox(norm(decoded_I - I(d)), 0, atol = tol * d * d)

    # test obs_noise_cov   (check the warning at the start)
    @test_logs (:warn,) (:info,) (:info,) (:warn,) Emulator(gp, io_pairs, obs_noise_cov = Σ)
    @test_logs (:warn,) (:info,) (:info,) (:warn,) Emulator(
        gp,
        io_pairs,
        input_structure_matrix = 4.0 * I,
        obs_noise_cov = 3.0 * I,
        output_structure_matrix = 2.0 * I,
    )
    em1 = Emulator(gp, io_pairs, obs_noise_cov = Σ)

    em2 = Emulator(
        gp,
        io_pairs,
        input_structure_matrix = 4.0 * I(p),
        obs_noise_cov = Σ,
        output_structure_matrix = 2.0 * I,
    )
    Γ = randn(p, p)
    Γ = Γ * Γ'
    em3 = Emulator(gp, io_pairs, input_structure_matrix = Γ)
    enc_sch1 = create_encoder_schedule([(decorrelate_sample_cov(), "in"), (decorrelate_structure_mat(), "out")])
    enc_sch2 = create_encoder_schedule([(decorrelate_structure_mat(), "in"), (decorrelate_structure_mat(), "out")])
    enc_sch3 = create_encoder_schedule([(decorrelate_structure_mat(), "in"), (decorrelate_sample_cov(), "out")])
    (_, _, _) = encode_with_schedule!(
        enc_sch1,
        io_pairs,
        1.0 * I(p),
        Σ, # obs noise cov becomes the output structure matrix
    )
    @test get_encoder_schedule(em1) == enc_sch1

    (_, _, _) = encode_with_schedule!(
        enc_sch2,
        io_pairs,
        4.0 * I(p),
        2.0 * I(d), # out_struct_mat overrides obs noise cov
    )
    @test get_encoder_schedule(em2) == enc_sch2

    (_, _, _) = encode_with_schedule!(enc_sch3, io_pairs, Γ, I(d))
    @test get_encoder_schedule(em3) == enc_sch3

end
