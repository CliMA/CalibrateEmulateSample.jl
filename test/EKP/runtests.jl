using Distributions
using LinearAlgebra
using Random
using Test
using Plots

using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Priors

@testset "EKP" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    ### Define priors on the parameters u
    umin = 0.0
    umax = 20.0
    dist1 = Uniform(umin, umax) # prior on u1
    dist2 = Uniform(umin, umax) # prior on u2
    priors = [Priors.Prior(dist1, "u1"), Priors.Prior(dist2, "u2")]
    prior_mean = [mean(dist1), mean(dist2)]
    # Assuming independence of u1 and u2
    prior_cov = convert(Array, Diagonal([var(dist1), var(dist2)]))

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
        truth_samples[i, :] = yt .+ Distributions.rand(MvNormal(μ, Γy))
    end

    ###
    ###  Calibrate (1): Ensemble Kalman Inversion
    ###

    N_ens = 50 # number of ensemble members
    N_iter = 5 # number of EKI iterations
    initial_params = EKP.construct_initial_ensemble(N_ens, priors;
                                                    rng_seed=rng_seed)
    ekiobj = EKP.EKObj(initial_params, param_names,
                       vec(mean(truth_samples, dims=1)), Γy, Inversion())

    @test size(initial_params) == (N_ens, npar)
    @test size(ekiobj.u[end])  == (N_ens, npar)
    @test ekiobj.unames == ["u1", "u2"]
    @test ekiobj.obs_mean ≈ [37.29, 15.49] atol=1e-1
    @test ekiobj.obs_noise_cov ≈ [30.29 0.0; 0.0 12.29] atol=1e-1
    @test ekiobj.N_ens == N_ens

    # test convergence warning during update
    params_i = ekiobj.u[end]
    g_ens = G(params_i)
    @test_logs (:warn, string("New ensemble covariance determinant is less than ",
                              "0.01 times its previous value.",
                              "\nConsider reducing the EK time step."))
    EKP.update_ensemble!(ekiobj, g_ens)

    # restart EKObj and test step
    ekiobj = EKP.EKObj(initial_params, param_names,
                       vec(mean(truth_samples, dims=1)), Γy, Inversion())
    @test EKP.find_ek_step(ekiobj, g_ens) ≈ 0.5

    # EKI iterations
    for i in 1:N_iter
        params_i = ekiobj.u[end]
        g_ens = G(params_i)
        EKP.update_ensemble!(ekiobj, g_ens)
    end

    # EKI results: Test if ensemble has collapsed toward the true parameter values
    eki_final_result = vec(mean(ekiobj.u[end], dims=1))
    @test norm(ut - eki_final_result) < 0.5


    ###
    ###  Calibrate (2): Ensemble Kalman Sampleer
    ###

    # EKS is not fully implemented yet, so this should raise an exception
    # @test_throws Exception eksobj = EKP.EKObj(initial_params, param_names,
    #                                           vec(mean(truth_samples, dims=1)),
    #                                           Γy, Sampler(prior_mean, prior_cov))

    # TODO: Uncomment L. 98-106 once EKS is implemented

    println(size(prior_mean))
    println(size(prior_cov))
    eksobj = EKP.EKObj(initial_params, param_names,
                       vec(mean(truth_samples, dims=1)), Γy,
                       Sampler(prior_mean, prior_cov))
    # EKS iterations
    println("Iterations: ", N_iter)
    for i in 1:N_iter
        params_i = eksobj.u[end]
        g_ens = G(params_i)
        EKP.update_ensemble!(eksobj, g_ens)
    end

   # Plot evolution of the EKI particles
   eki_final_result = vec(mean(ekiobj.u[end], dims=1))
   gr()
   p = plot(ekiobj.u[1][:,1], ekiobj.u[1][:,2], seriestype=:scatter)
   for i in 2:N_iter
       plot!(ekiobj.u[i][:, 1],  ekiobj.u[i][:,2], seriestype=:scatter)
   end
   plot!([ut[1]], xaxis="u1", yaxis="u2", seriestype="vline",
         linestyle=:dash, linecolor=:red)
   plot!([ut[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
   savefig(p, "EKI_test.png")

   # Plot evolution of the EKS particles
   eks_final_result = vec(mean(eksobj.u[end], dims=1))
   gr()
   p = plot(eksobj.u[1][:,1], eksobj.u[1][:,2], seriestype=:scatter)
   for i in 2:N_iter
       plot!(eksobj.u[i][:, 1],  eksobj.u[i][:,2], seriestype=:scatter)
   end
   plot!([ut[1]], xaxis="u1", yaxis="u2", seriestype="vline",
         linestyle=:dash, linecolor=:red)
   plot!([ut[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
   savefig(p, "EKS_test.png")

    #@test norm(ut - eks_final_result) < 0.5
end
