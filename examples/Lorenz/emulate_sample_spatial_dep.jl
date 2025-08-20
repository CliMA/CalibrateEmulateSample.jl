# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using StatsPlots
using Plots
using Random
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers


function main()

    cases = [
        "GP",
        "RF-scalar", # diagonalize, train scalar RF, don't asume diag inputs
    ]

    #### CHOOSE YOUR CASE: 
    mask = [1]# 1:1 # e.g. 1:2 or [2]
    for (case) in cases[mask]


        println("case: ", case)
        min_iter = 1
        max_iter = 7 # number of EKP iterations to use data from is at most this

        ####

        exp_name = "Lorenz_histogram_spatial_dep"
        rng_seed = 44011
        rng = Random.MersenneTwister(rng_seed)

        # loading relevant data
        homedir = pwd()
        println(homedir)
        figure_save_directory = joinpath(homedir, "output/")
        data_save_directory = joinpath(homedir, "output/")
        data_save_files = [
            joinpath(data_save_directory, "ekp_spatial_dep.jld2"),
            joinpath(data_save_directory, "priors_spatial_dep.jld2"),
            joinpath(data_save_directory, "calibrate_results_spatial_dep.jld2"),
        ]
        if !isfile(data_save_files[1])
            throw(ErrorException("data files not found. \n First run: \n > julia --project calibrate_spatial_dep.jl"))
        end

        ekpobj = load(data_save_files[1])["ekpobj"]
        priors = load(data_save_files[2])["prior"]
        truth_sample = load(data_save_files[3])["truth_sample"]
        truth_params_constrained = load(data_save_files[3])["truth_input_constrained"] #true parameters in constrained space
        truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)
        Γy = get_obs_noise_cov(ekpobj)

        n_params = length(truth_params) # "input dim"
        output_dim = size(Γy, 1)
        ###
        ###  Emulate: Gaussian Process Regression
        ###

        # Emulate-sample settings
        # choice of machine-learning tool in the emulation stage
        nugget = 1e-3
        if case == "GP"
            gppackage = Emulators.GPJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(
                gppackage;
                kernel = nothing, # use default squared exponential kernel
                prediction_type = pred_type,
                noise_learn = false,
            )
        elseif case == "RF-scalar"
            overrides = Dict(
                #       "verbose" => true,
                "scheduler" => DataMisfitController(terminate_at = 1000.0),
                "cov_sample_multiplier" => 1.0,
                "n_iteration" => 8,
                "n_features_opt" => 40,
            )
            n_features = 80
            kernel_structure = SeparableKernel(LowRankFactor(1, nugget), OneDimFactor())
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_params,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = overrides,
            )
        else
            throw(ArgumentError("case $(case) not recognised, please choose from the list $(cases)"))
        end


        # Get training points from the EKP iteration number in the second input term  
        N_iter = min(max_iter, length(get_u(ekpobj)) - 1) # number of paired iterations taken from EKP
        min_iter = min(max_iter, max(1, min_iter))
        input_output_pairs = Utilities.get_training_points(ekpobj, min_iter:(N_iter - 1))
        input_output_pairs_test = Utilities.get_training_points(ekpobj, N_iter:(length(get_u(ekpobj)) - 1)) #  "next" iterations
        # Save data
        @save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

        # plot training points in constrained space
        if case == cases[mask[1]]
            i1 = 1
            i2 = 10
            gr(dpi = 300, size = (400, 400))
            inputs_unconstrained = get_inputs(input_output_pairs)
            inputs_constrained = transform_unconstrained_to_constrained(priors, inputs_unconstrained)
            p = plot(
                title = "training points",
                xlims = extrema(inputs_constrained[i1, :]),
                xaxis = "x$(i1)",
                yaxis = "x$(i2)",
                ylims = extrema(inputs_constrained[i2, :]),
            )
            scatter!(p, inputs_constrained[i1, :], inputs_constrained[i2, :], color = :magenta, label = "train")
            inputs_test_unconstrained = get_inputs(input_output_pairs_test)
            inputs_test_constrained = transform_unconstrained_to_constrained(priors, inputs_test_unconstrained)

            scatter!(p, inputs_test_constrained[i1, :], inputs_test_constrained[i2, :], color = :black, label = "test")

            vline!(p, [truth_params_constrained[i1]], linestyle = :dash, linecolor = :red, label = false)
            hline!(p, [truth_params_constrained[i2]], linestyle = :dash, linecolor = :red, label = false)
            savefig(p, joinpath(figure_save_directory, "training_points.pdf"))
            savefig(p, joinpath(figure_save_directory, "training_points.png"))
        end

        # data processing configuration
        retain_var = 0.95
        encoder_schedule = [(decorrelate_structure_mat(retain_var = retain_var), "in_and_out")]

        emulator = Emulator(
            mlt,
            input_output_pairs,
            (; prior_cov = cov(priors), obs_noise_cov = Γy);
            encoder_schedule = encoder_schedule,
        )
        optimize_hyperparameters!(emulator)

        # Check how well the Gaussian Process regression predicts on the
        # true parameters
        y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1), transform_to_real = true)
        y_mean_test, y_var_test =
            Emulators.predict(emulator, get_inputs(input_output_pairs_test), transform_to_real = true)

        println("ML prediction on true parameters: ")
        println(vec(y_mean))
        println("true data: ")
        println(truth_sample) # what was used as truth
        println(" ML predicted standard deviation")
        println(sqrt.(diag(y_var[1], 0)))
        println("ML MSE (truth): ")
        println(mean((truth_sample - vec(y_mean)) .^ 2))
        println("ML MSE (next ensemble): ")
        println(mean((get_outputs(input_output_pairs_test) - y_mean_test) .^ 2))

        #end
        ###
        ###  Sample: Markov Chain Monte Carlo
        ###
        # initial values
        u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
        println("initial parameters: ", u0)

        # First let's run a short chain to determine a good step size
        mcmc = MCMCWrapper(RWMHSampling(), truth_sample, priors, emulator; init_params = u0)
        new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

        # Now begin the actual MCMC
        println("Begin MCMC - with step size ", new_step)
        chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)

        posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

        post_mean = mean(posterior)
        post_cov = cov(posterior)
        println("post_mean")
        println(post_mean)
        println("post_cov")
        println(post_cov)
        println("D util")
        println(det(inv(post_cov)))
        println(" ")

        param_names = get_name(posterior)

        posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
        constrained_posterior_samples =
            mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

        # marginal histogram
        pp = plot(priors, c = :gray)
        plot!(pp, posterior)
        final_params_constrained = get_ϕ_mean_final(priors, ekpobj)
        for (i, sp) in enumerate(pp.subplots)
            vline!(sp, [truth_params_constrained[i]], lc = :black, lw = 4)
            vline!(sp, [final_params_constrained[i]], lc = :magenta, lw = 4)
        end
        figpath = joinpath(figure_save_directory, "posterior_hist_" * case)
        savefig(figpath * ".png")
        savefig(figpath * ".pdf")

        # Save data
        save(
            joinpath(data_save_directory, "posterior_$(case).jld2"),
            "posterior",
            posterior,
            "priors",
            priors,
            "truth_params_constrained",
            truth_params_constrained,
            "final_params_constrained",
            final_params_constrained,
            "input_output_pairs",
            input_output_pairs,
            "truth_params",
            truth_params,
            "encoder_schedule",
            get_encoder_schedule(emulator),
        )
    end
end

main()
