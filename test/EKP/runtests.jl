using Distributions
using LinearAlgebra
using Random
using Test

using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Priors

@testset "EKP" begin

    # Seed for pseudo-random number generator
    rng_seed = 41
    Random.seed!(rng_seed)

    ### Generate data from a linear model: a regression problem with n_par parameters
    ### and n_obs. G(u) = A \times u, where A : R^n_par -> R
    n_obs = 10                  # Number of synthetic observations from G(u)
    n_par =  2                  # Number of parameteres
    u_star = [-1., 2.]          # True parameters
    noise_level = .1            # Defining the observation noise level
    Γy = noise_level * Matrix(I, n_obs, n_obs) # Independent noise for synthetic observations
    noise = MvNormal(zeros(n_obs), Γy)

    C = [1 -.9; -.9 1]          # Correlation structure for linear operator
    A = rand(MvNormal(zeros(2,), C), n_obs)'    # Linear operator in R^{n_obs \times n_par}

    @test size(A) == (n_obs, n_par)

    y_star = A * u_star
    y_obs  = y_star + rand(noise)

    @test size(y_star) == (n_obs,)

    #### Define linear model
    function G(u)
        A * u
    end

    @test norm(y_star - G(u_star)) < n_obs * noise_level^2

    #### Define prior information on parameters
    param_names = ["u1", "u2"]
    priors     = [Priors.Prior(Normal(1., sqrt(2)), param_names[1]),
                  Priors.Prior(Normal(1., sqrt(2)), param_names[2])]

    prior_mean = [1., 1.]
    # Assuming independence of u1 and u2
    prior_cov = convert(Array, Diagonal([sqrt(2.), sqrt(2.)]))

    ###
    ###  Calibrate (1): Ensemble Kalman Inversion
    ###

    N_ens = 50 # number of ensemble members
    N_iter = 20 # number of EKI iterations
    initial_ensemble = EKP.construct_initial_ensemble(N_ens, priors;
                                                    rng_seed=rng_seed)
    @test size(initial_ensemble) == (N_ens, n_par)

    ekiobj = EKP.EKObj(initial_ensemble, param_names,
                       y_obs, Γy, Inversion())

    # EKI iterations
    for i in 1:N_iter
        params_i = ekiobj.u[end]
        g_ens = G(params_i')'
        EKP.update_ensemble!(ekiobj, g_ens)
    end

    # EKI results: Test if ensemble has collapsed toward the true parameter 
    # values
    eki_final_result = vec(mean(ekiobj.u[end], dims=1))
    # @test norm(u_star - eki_final_result) < 0.5

    # Plot evolution of the EKI particles
    eki_final_result = vec(mean(ekiobj.u[end], dims=1))
    
    if TEST_PLOT_OUTPUT
        gr()
        p = plot(ekiobj.u[1][:,1], ekiobj.u[1][:,2], seriestype=:scatter)
        plot!(ekiobj.u[end][:, 1],  ekiobj.u[end][:,2], seriestype=:scatter)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red)
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
        savefig(p, "EKI_test.png")
    end

    ###
    ###  Calibrate (2): Ensemble Kalman Sampleer
    ###
    eksobj = EKP.EKObj(initial_ensemble, param_names,
                      y_obs, Γy,
                      Sampler(prior_mean, prior_cov))

    # EKS iterations
    for i in 1:N_iter
        params_i = eksobj.u[end]
        g_ens = G(params_i')'
        EKP.update_ensemble!(eksobj, g_ens)
    end

    # Plot evolution of the EKS particles
    eks_final_result = vec(mean(eksobj.u[end], dims=1))
    
    if TEST_PLOT_OUTPUT
        gr()
        p = plot(eksobj.u[1][:,1], eksobj.u[1][:,2], seriestype=:scatter)
        plot!(eksobj.u[end][:, 1],  eksobj.u[end][:,2], seriestype=:scatter)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red)
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
        savefig(p, "EKS_test.png")
    end

    posterior_cov_inv = (A' * (Γy\A) + 1 * Matrix(I, n_par, n_par)/prior_cov)
    ols_mean          = (A'*(Γy\A)) \ (A'*(Γy\y_obs))
    posterior_mean    = posterior_cov_inv \ ( (A'*(Γy\A)) * ols_mean  + (prior_cov\prior_mean))

    #### This tests correspond to:
    # EKI provides a solution closer to the ordinary Least Squares estimate
    @test norm(ols_mean - eki_final_result) < norm(ols_mean - eks_final_result)
    # EKS provides a solution closer to the posterior mean
    @test norm(posterior_mean - eks_final_result) < norm(posterior_mean - eki_final_result)

    ##### I expect this test to make sense:
    # In words: the ensemble covariance is still a bit ill-dispersed since the
    # algorithm employed still does not include the correction term for finite-sized
    # ensembles.
    @test abs(sum(diag(posterior_cov_inv\cov(eksobj.u[end]))) - n_par) > 1e-5
end
