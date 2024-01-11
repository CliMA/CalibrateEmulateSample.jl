using CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions

function calibrate_model(observations;
                         temperature_scale_factor = 5,
                         OLR_scale_factor = 1,
                         ASR_scale_factor = 1,
                         N_ens = 30,    # number of ensemble members
                         N_iter = 20)   # number of EKI iterations


    # Construct observation object
    truth = Observations.Observation(yt, Γy, data_names)
    truth_sample = yt[:, end]

    #####
    #####  Calibrate: Ensemble Kalman Inversion
    #####

    # EKP parameters
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
        g_ens = GModel.run_ensemble(params_i, lorenz_settings_G)
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