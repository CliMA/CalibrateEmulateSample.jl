using Distributions
using LinearAlgebra
using Random
using Test

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributionStorage
@testset "EnsembleKalmanProcesses" begin

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
    prior_distns = [Parameterized(Normal(1., sqrt(2))),
              Parameterized(Normal(1., sqrt(2)))]
    constraints = [[no_constraint()], [no_constraint()]]
    prior_names = ["u1","u2"]
    prior = ParameterDistribution(prior_distns, constraints, prior_names)
    
    prior_mean = reshape(get_mean(prior),:)

    # Assuming independence of u1 and u2
    prior_cov = get_cov(prior)#convert(Array, Diagonal([sqrt(2.), sqrt(2.)]))

    ###
    ###  Calibrate (1): Ensemble Kalman Inversion
    ###

    N_ens = 50 # number of ensemble members
    N_iter = 20 # number of EKI iterations
    initial_ensemble = EnsembleKalmanProcesses.construct_initial_ensemble(prior,N_ens;
                                                    rng_seed=rng_seed)
    @test size(initial_ensemble) == (n_par, N_ens)

    ekiobj = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_ensemble,
                       y_obs, Γy, Inversion())

    # Find EKI timestep
    g_ens = G(get_u_final(ekiobj))
    @test size(g_ens) == (n_obs, N_ens)
    # as the columns of g are the data, this should throw an error
    g_ens_t = permutedims(g_ens, (2,1))
    @test_throws DimensionMismatch find_ekp_stepsize(ekiobj, g_ens_t)
    Δ = find_ekp_stepsize(ekiobj, g_ens)
    @test Δ ≈ 0.0625
    
    # EKI iterations
    for i in 1:N_iter
        params_i = get_u_final(ekiobj)
        g_ens = G(params_i)
        EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens)
    end

    # EKI results: Test if ensemble has collapsed toward the true parameter 
    # values
    eki_final_result = vec(mean(get_u_final(ekiobj), dims=2))
    # @test norm(u_star - eki_final_result) < 0.5

    # Plot evolution of the EKI particles
    eki_final_result = vec(mean(get_u_final(ekiobj), dims=2))
    
    if TEST_PLOT_OUTPUT
        gr()
        p = plot(get_u_prior(ekiobj)[1,:], get_u_prior(ekiobj)[2,:], seriestype=:scatter)
        plot!(get_u_final(ekiobj)[1, :],  get_u_final(ekiobj)[2,:], seriestype=:scatter)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red)
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red)
        savefig(p, "EKI_test.png")
    end

    ###
    ###  Calibrate (2): Ensemble Kalman Sampleer
    ###
    eksobj = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_ensemble,
                      y_obs, Γy,
                      Sampler(prior_mean, prior_cov))

    # EKS iterations
    for i in 1:N_iter
        params_i = get_u_final(eksobj)
        g_ens = G(params_i)
        EnsembleKalmanProcesses.update_ensemble!(eksobj, g_ens)
    end

    # Plot evolution of the EKS particles
    eks_final_result = vec(mean(get_u_final(eksobj), dims=2))
    
    if TEST_PLOT_OUTPUT
        gr()
        p = plot(get_u_prior(eksobj)[1,:], get_u_prior(eksobj)[2,:], seriestype=:scatter)
        plot!(get_u_final(eksobj)[1, :],  get_u_final(eksobj)[2,:], seriestype=:scatter)
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
    @test abs(sum(diag(posterior_cov_inv\cov(get_u_final(eksobj),dims=2))) - n_par) > 1e-5
end
