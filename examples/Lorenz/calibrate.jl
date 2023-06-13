### This script is copied from Lorenz 96 example in EnsembleKalmanProcesses.jl v0.14.2

include("GModel.jl") # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2

# CES
using CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions

function main()

    rng_seed = 4137
    rng = Random.MersenneTwister(rng_seed)
    # Output figure save directory
    homedir = pwd()
    println(homedir)
    figure_save_directory = homedir * "/output/"
    data_save_directory = homedir * "/output/"
    if !isdir(figure_save_directory)
        mkdir(figure_save_directory)
    end
    if !isdir(data_save_directory)
        mkdir(data_save_directory)
    end

    # Governing settings
    # Characteristic time scale
    τc = 5.0 # days, prescribed by the L96 problem
    # Stationary or transient dynamics
    dynamics = 2 # Transient is 2
    # Statistics integration length
    # This has to be less than 360 and 360 must be divisible by Ts_days
    Ts_days = 30.0 # Integration length in days (will be made non-dimensional later)
    # Stats type, which statistics to construct from the L96 system
    # 4 is a linear fit over a batch of length Ts_days
    stats_type = 5
    # 5 is the mean over batches of length Ts_days:


    ###
    ###  Define the (true) parameters
    ###
    # Define the parameters that we want to learn
    F_true = 8.0 # Mean F
    A_true = 2.5 # Transient F amplitude
    ω_true = 2.0 * π / (360.0 / τc) # Frequency of the transient F (non-dim)

    if dynamics == 2
        params_true = [F_true, A_true]
        param_names = ["F", "A"]
    else
        params_true = [F_true]
        param_names = ["F"]
    end
    n_param = length(param_names)
    params_true = reshape(params_true, (n_param, 1))

    println(n_param)
    println(params_true)


    ###
    ###  Define the parameter priors
    ###

    if dynamics == 2
        prior_means = [F_true + 1.0, A_true + 0.5]
        prior_stds = [2.0, 0.5 * A_true]
        prior_F = PD.constrained_gaussian(param_names[1], prior_means[1], prior_stds[1], 0, Inf)
        prior_A = PD.constrained_gaussian(param_names[2], prior_means[2], prior_stds[2], 0, Inf)
        priors = PD.combine_distributions([prior_F, prior_A])
    else
        priors = PD.constrained_gaussian("F", F_true, 1.0, 0, Inf)
    end


    ###
    ###  Define the data from which we want to learn the parameters
    ###
    data_names = ["y0", "y1"]


    ###
    ###  L96 model settings
    ###

    # Lorenz 96 model parameters
    # Behavior changes depending on the size of F, N
    N = 36
    dt = 1 / 64.0
    # Start of integration
    t_start = 800.0
    # We now rescale all the timings etc, to be in nondim-time
    # Data collection length
    if dynamics == 1
        T = 2.0
    else
        T = 360.0 / τc
    end
    # Batch length
    Ts = 5.0 / τc # Nondimensionalize by L96 timescale
    # Integration length
    Tfit = Ts_days / τc
    # Initial perturbation
    Fp = rand(Normal(0.0, 0.01), N)
    kmax = 1
    # Prescribe variance or use a number of forward passes to define true interval variability
    var_prescribe = false
    # Use CES to learn ω?
    ω_fixed = true

    # Note, the initial conditions are given by Fp, and the integration is run from this
    # one can change the ICs of a simulation therefore by changing Fp
    # Constructs an LSettings structure, see GModel.jl for the descriptions
    lorenz_settings =
        GModel.LSettings(dynamics, stats_type, t_start, T, Ts, Tfit, Fp, N, dt, t_start + T, kmax, ω_fixed, ω_true)
    println(lorenz_settings)
    lorenz_params = GModel.LParams(F_true, ω_true, A_true)

    ###
    ###  Generate (artificial) truth samples
    ###  Note: The observables y are related to the parameters θ by:
    ###        y = G(θ) + η
    ###

    # Lorenz forward
    # Input: params: [N_params, N_ens]
    # Output: gt: [N_data, N_ens]
    # Dropdims of the output since the forward model is only being run with N_ens=1 
    # corresponding to the truth construction
    gt = dropdims(GModel.run_G_ensemble(params_true, lorenz_settings), dims = 2)

    # Compute internal variability covariance
    if var_prescribe == true
        n_samples = 100
        yt = zeros(length(gt), n_samples)
        noise_sd = 0.25
        #        Γy = noise_sd^2 * convert(Array, Diagonal(gt))
        Γy = noise_sd^2 * convert(Array, Diagonal(ones(length(gt)) * mean(gt)))
        μ = zeros(length(gt))
        # Add noise
        for i in 1:n_samples
            yt[:, i] = gt .+ rand(MvNormal(μ, Γy))
        end
        println(Γy)

    else
        println("Using truth values to compute covariance")
        n_samples = 100
        nthreads = Threads.nthreads()
        yt = zeros(nthreads, length(gt), n_samples)
        Threads.@threads for i in 1:n_samples

            #=
                        lorenz_settings_local = GModel.LSettings(
                            dynamics,
                            stats_type,
                            t_start + T * (i - 1),
                            T,
                            Ts,
                            Tfit,
                            Fp,
                            N,
                            dt,
                            t_start + T * (i - 1) + T,
                            kmax,
                            ω_fixed,
                            ω_true,
                        )=#
            lorenz_settings_local = lorenz_settings
            tid = Threads.threadid()
            yt[tid, :, i] = GModel.run_G_ensemble(params_true, lorenz_settings_local)
        end
        yt = dropdims(sum(yt, dims = 1), dims = 1)

        # add a little extra variance for regularization here
        noise_sd = 0.1
        Γy_diag = noise_sd .^ 2 * convert(Array, I(size(yt, 1)))
        μ = zeros(size(yt, 1))
        # Add noise
        yt += rand(MvNormal(μ, Γy_diag), size(yt, 2))

        # Variance of truth data
        #Γy = convert(Array, Diagonal(dropdims(mean((yt.-mean(yt,dims=1)).^2,dims=1),dims=1)))
        # Covariance of truth data
        Γy = cov(yt, dims = 2)

        println(Γy)
    end



    # Construct observation object
    truth = Observations.Observation(yt, Γy, data_names)
    truth_sample = yt[:, end]
    ###
    ###  Calibrate: Ensemble Kalman Inversion
    ###

    # L96 settings for the forward model in the EKP. Here, the forward model for the EKP settings can be set distinctly from the truth runs
    lorenz_settings_G = lorenz_settings # initialize to truth settings

    # EKP parameters
    N_ens = 30 # number of ensemble members
    N_iter = 20 # number of EKI iterations
    # initial parameters: N_params x N_ens
    initial_params = EKP.construct_initial_ensemble(rng, priors, N_ens)

    ekiobj = EKP.EnsembleKalmanProcess(
        initial_params,
        truth_sample,
        truth.obs_noise_cov,
        EKP.Inversion(),
        scheduler = EKP.DataMisfitController(),
        verbose = true,
    )

    # EKI iterations
    println("EKP inversion error:")
    err = zeros(N_iter)
    final_iter = [N_iter]
    for i in 1:N_iter
        params_i = EKP.get_ϕ_final(priors, ekiobj) # the `ϕ` indicates that the `params_i` are in the constrained space    
        g_ens = GModel.run_G_ensemble(params_i, lorenz_settings_G)
        terminated = EKP.update_ensemble!(ekiobj, g_ens)
        if !isnothing(terminated)
            final_iter = i - 1 # final update was previous iteration
            break
        end
        err[i] = EKP.get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
        println("Iteration: " * string(i) * ", Error: " * string(err[i]))
    end
    N_iter = final_iter[1] #in case it terminated early

    # EKI results: Has the ensemble collapsed toward the truth?
    println("True parameters: ")
    println(params_true)

    println("\nEKI results:")
    println(EKP.get_ϕ_mean_final(priors, ekiobj))

    u_stored = EKP.get_u(ekiobj, return_array = false)
    g_stored = EKP.get_g(ekiobj, return_array = false)

    save(
        joinpath(data_save_directory, "calibrate_results.jld2"),
        "inputs",
        u_stored,
        "outputs",
        g_stored,
        "priors",
        priors,
        "eki",
        ekiobj,
        "truth_sample",
        truth_sample,
        "truth_sample_mean",
        truth.mean,
        "truth_input_constrained",
        params_true, #constrained here, as these are in a physically constrained space (unlike the u inputs),
    )

    # plots in constrained space
    gr(dpi = 300, size = (400, 400))
    ϕ_init = EKP.get_ϕ(priors, ekiobj, 1)
    p = plot(
        title = "EKI iterations",
        xlims = extrema(ϕ_init[1, :]),
        xaxis = "F",
        yaxis = "A",
        ylims = extrema(ϕ_init[2, :]),
    )
    for i in 1:N_iter
        ϕ_i = EKP.get_ϕ(priors, ekiobj, i)
        scatter!(p, ϕ_i[1, :], ϕ_i[2, :], color = :grey, label = false)
    end
    ϕ_fin = EKP.get_ϕ_final(priors, ekiobj)
    scatter!(p, ϕ_fin[1, :], ϕ_fin[2, :], color = :magenta, label = false)

    vline!(p, [params_true[1]], linestyle = :dash, linecolor = :red, label = false)
    hline!(p, [params_true[2]], linestyle = :dash, linecolor = :red, label = false)
    savefig(p, joinpath(figure_save_directory, "calibration.png"))
    savefig(p, joinpath(figure_save_directory, "calibration.pdf"))
end

main()
