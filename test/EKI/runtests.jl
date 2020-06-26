using Distributions
using LinearAlgebra
using Random
using Test

using CalibrateEmulateSample.EKI
using CalibrateEmulateSample.Priors

@testset "EKI" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    ### Define priors on the parameters u
    umin = 0.0
    umax = 20.0
    priors = [Priors.Prior(Uniform(umin, umax), "u1"),  # prior on u1
              Priors.Prior(Uniform(umin, umax), "u2")]  # prior on u2

    ### Define forward model G
    function G(u)
        3.0 .* u
    end

    ###  Generate (artificial) truth samples
    npar = 2 # number of parameters
    ut = Distributions.rand(Uniform(umin, umax), npar)
    param_names = ["u1", "u2"]
    yt = G(ut)
    n_samples = 100
    truth_samples = zeros(n_samples, length(yt))
    noise_level = 0.9
    Γy = noise_level^2 * convert(Array, Diagonal(yt))
    μ = zeros(length(yt))
    for i in 1:n_samples
        truth_samples[i, :] = yt + noise_level^2 * Distributions.rand(MvNormal(μ, Γy))
    end

    ###  Calibrate: Ensemble Kalman Inversion
    N_ens = 50 # number of ensemble members
    N_iter = 5 # number of EKI iterations
    initial_params = EKI.construct_initial_ensemble(N_ens, priors; 
                                                    rng_seed=rng_seed)
    ekiobj = EKI.EKIObj(initial_params, param_names, 
                        vec(mean(truth_samples, dims=1)), Γy)

    @test size(initial_params) == (N_ens, npar)
    @test size(ekiobj.u[end])  == (N_ens, npar)
    @test ekiobj.unames == ["u1", "u2"]
    @test ekiobj.g_t ≈ [37.29, 15.49] atol=1e-1
    @test ekiobj.cov ≈ [30.29 0.0; 0.0 12.29] atol=1e-1
    @test ekiobj.N_ens == N_ens

    # test convegence warning during update
    params_i = ekiobj.u[end]
    g_ens = G(params_i)
    @test_logs (:warn,"Ensemble covariance after the last EKI stage decreased significantly. 
            Consider adjusting the EKI time step.") EKI.update_ensemble!(ekiobj, g_ens)

    # restart EKIObj and test step
    ekiobj = EKI.EKIObj(initial_params, param_names, 
                        vec(mean(truth_samples, dims=1)), Γy)
    @test EKI.find_eki_step(ekiobj, g_ens) ≈ 0.5

    # EKI iterations
    for i in 1:N_iter
        params_i = ekiobj.u[end]
        g_ens = G(params_i)
        EKI.update_ensemble!(ekiobj, g_ens) 
    end

    # EKI results: Test if ensemble has collapsed toward the true parameter values
    eki_final_result = vec(mean(ekiobj.u[end], dims=1))
    @test norm(ut - eki_final_result) < 0.5
end
