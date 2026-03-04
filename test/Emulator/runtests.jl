
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
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions

const PD = ParameterDistributions
const EM = Emulators
#build an unknown type
struct MLTester <: Emulators.MachineLearningTool end

@testset "Emulators" begin
    #build some quick data + noise
    m = 50
    d = 6
    p = 10
    x = rand(p, m) 
    g = randn(d, p) 
    y = g*x
    
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
    enc_io_pairs, enc_I_in, enc_I_out = initialize_and_encode_with_schedule!(enc_sch, io_pairs; obs_noise_cov = 1.0 * I)
    # NB this gives encoder up to sign
    tol = 1e-12
    @test get_encoder_schedule(em) == enc_sch # inputs: proc
    @test all(isapprox.(get_inputs(get_encoded_io_pairs(em)), get_inputs(enc_io_pairs), atol = tol))
    @test all(isapprox.(get_outputs(get_encoded_io_pairs(em)), get_outputs(enc_io_pairs), atol = tol))
    @test isempty(enc_I_in)

    #NB - encoders all tested in Utilities here just testing some API
    encoded_mat = encode_data(em, x, "in")
    encoded_dc = encode_data(em, DataContainer(x), "in")
    encoded_I = encode_structure_matrix(em, 1.0 * I, "out")
    tol = 1e-14
    @test isapprox(norm(encoded_mat - get_data(encoded_dc)), 0, atol = tol * p * m)
    @test isapprox(norm(encoded_mat - get_inputs(enc_io_pairs)), 0, atol = tol * p * m)
    @test isapprox(norm(enc_I_out[:obs_noise_cov] - encoded_I), 0, atol = tol * d * d)

    decoded_dc = decode_data(em, encoded_dc, "in")
    decoded_mat = decode_data(em, encoded_mat, "in")
    decoded_I = decode_structure_matrix(em, encoded_I, "out")
    @test isapprox(norm(get_data(decoded_dc) - decoded_mat), 0, atol = tol * p * m)
    @test isapprox(norm(decoded_mat - x), 0, atol = tol * p * m)
    @test isapprox(norm(decoded_I - 1.0 * I), 0, atol = tol * d * d)

    # test obs_noise_cov   (check the warning at the start)
    @test_logs (:warn,) (:info,) (:warn,) (:info,) (:warn,) Emulator(gp, io_pairs, obs_noise_cov = Σ)
    @test_logs (:warn,) (:info,) (:warn,) (:info,) (:warn,) Emulator(
        gp,
        io_pairs;
        encoder_kwargs = (; prior_cov = 4.0 * I, obs_noise_cov = 2.0 * I),
        obs_noise_cov = 3.0 * I,
    )
    em1 = Emulator(gp, io_pairs; encoder_kwargs = (; obs_noise_cov = Σ))

    enc_sch1 = create_encoder_schedule([(decorrelate_sample_cov(), "in"), (decorrelate_structure_mat(), "out")])
    initialize_and_encode_with_schedule!(
        enc_sch1,
        io_pairs;
        prior_cov = 1.0 * I(p),
        obs_noise_cov = Σ, # obs noise cov becomes the output structure matrix
    )
    @test get_encoder_schedule(em1) == enc_sch1

end

@testset "Emulators" begin
    #build some quick data + noise
    m = 50
    d = 6
    p = 10
    prior = constrained_gaussian("10d_pos", 1, 0.5, 0, Inf, repeats=p)  
    x = PD.sample(prior,m) # p x m (sampled in unconstrained space)
    g = randn(d, p)
    G(x) = g*log.(x)  # can only be applied to positive constrained x
    y = reduce(hcat, G(transform_unconstrained_to_constrained(prior, xcol)) for xcol in eachcol(x)) # d x m
    
    # "noise"
    μ = zeros(d)
    Σ = rand(d, d)
    Σ = Σ' * Σ
    noise_samples = rand(MvNormal(μ, Σ), m)
    y += noise_samples

    io_pairs = PairedDataContainer(x, y, data_are_columns = true)
    
    # Test forward map wrapper with default encoding
    fmw = forward_map_wrapper(G,prior,io_pairs)
    @test get_forward_map(fmw) == G
    @test get_prior(fmw) == prior
    @test get_io_pairs(fmw) == io_pairs
    
    default_encoder = (decorrelate_sample_cov(), "in_and_out") # for these inputs this is the default
    enc_sch = create_encoder_schedule(default_encoder)
    enc_io_pairs, enc_I_in, enc_I_out = initialize_and_encode_with_schedule!(enc_sch, io_pairs; obs_noise_cov = 1.0 * I)
    tol = 1e-14
    @test get_encoder_schedule(fmw) == enc_sch # inputs: proc
    @test all(isapprox.(get_inputs(get_encoded_io_pairs(fmw)), get_inputs(enc_io_pairs), atol = tol))
    @test all(isapprox.(get_outputs(get_encoded_io_pairs(fmw)), get_outputs(enc_io_pairs), atol = tol))
    @test isempty(enc_I_in)

    # test some predictons
    x_test = PD.sample(prior, m) 
    y_test = reduce(hcat,G(transform_unconstrained_to_constrained(prior, xcol)) for xcol in eachcol(x_test))

    y_pred, y_cov = EM.predict(fmw, x_test; transform_to_real = true) 
    @test all(isapprox(norm(y_pred-y_test),0; atol=sqrt(d*m)*tol))
    sample_Σ = decode_structure_matrix(fmw, I, "out")
    @test all(isapprox(norm(sample_Σ-yc),0; atol=d*tol) for yc in y_cov)

    y_pred_enc, y_cov_enc = EM.predict(fmw, x_test; transform_to_real = false) 
    y_test_enc = encode_data(fmw, y_test, "out")
    @test all(isapprox(norm(y_pred_enc-y_test_enc),0; atol=sqrt(d*m)*tol))
    @test_throws ArgumentError EM.predict(fmw, x_test'; transform_to_real = true) 
    
end
