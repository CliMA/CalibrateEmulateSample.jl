# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using Plots
using Random
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

function main()

    cases = [
        "gp-gpjl", # diagonalize, train scalar GP, assume diag inputs
        "rf-lr-scalar",
        "rf-lr-lr",
        "rf-nonsep",
    ]

    #### CHOOSE YOUR CASE: 
    mask = 1:4 # 
    for (case) in cases[mask]


        println("case: ", case)
        min_iter = 1
        max_iter = 5 # number of EKP iterations to use data from is at most this
        overrides = Dict(
            "scheduler" => DataMisfitController(terminate_at = 100.0),
            "cov_sample_multiplier" => 1.0,
            "n_iteration" => 20,
        )
        # we do not want termination, as our priors have relatively little interpretation

        # Should be loaded:
        τc = 5.0
        F_true = 8.0 # Mean F
        A_true = 2.5 # Transient F amplitude
        ω_true = 2.0 * π / (360.0 / τc) # Frequency of the transient F
        ####

        exp_name = "Lorenz_histogram_F$(F_true)_A$(A_true)-w$(ω_true)"
        rng_seed = 44011
        rng = Random.MersenneTwister(rng_seed)

        # loading relevant data
        homedir = pwd()
        println(homedir)
        figure_save_directory = joinpath(homedir, "output/")
        data_save_directory = joinpath(homedir, "output/")
        data_save_file = joinpath(data_save_directory, "calibrate_results.jld2")

        if !isfile(data_save_file)
            throw(
                ErrorException(
                    "data file $data_save_file not found. \n First run: \n > julia --project calibrate.jl \n and store results $data_save_file",
                ),
            )
        end

        ekiobj = load(data_save_file)["eki"]
        priors = load(data_save_file)["priors"]
        truth_sample_mean = load(data_save_file)["truth_sample_mean"]
        truth_sample = load(data_save_file)["truth_sample"]
        truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
        truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)
        Γy = get_obs_noise_cov(ekiobj)


        n_params = length(truth_params) # "input dim"
        output_dim = size(Γy, 1)
        ###
        ###  Emulate: Gaussian Process Regression
        ###

        # Emulate-sample settings
        # choice of machine-learning tool in the emulation stage
        nugget = 0.001
        if case == "gp-gpjl"
            gppackage = Emulators.GPJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(
                gppackage;
                kernel = nothing, # use default squared exponential kernel
                prediction_type = pred_type,
                noise_learn = false,
            )
        elseif case == "rf-lr-scalar"
            n_features = 100
            kernel_structure = SeparableKernel(LowRankFactor(1, nugget), OneDimFactor())
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_params,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = overrides,
            )
        elseif case == "rf-lr-lr"
            # do we want to assume that the outputs are decorrelated in the machine-learning problem?
            kernel_structure = SeparableKernel(LowRankFactor(2, nugget), LowRankFactor(2, nugget))
            n_features = 500

            mlt = VectorRandomFeatureInterface(
                n_features,
                n_params,
                output_dim,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = overrides,
            )
        elseif case == "rf-nonsep"
            kernel_structure = NonseparableKernel(LowRankFactor(3, nugget))
            n_features = 500

            mlt = VectorRandomFeatureInterface(
                n_features,
                n_params,
                output_dim,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = overrides,
            )
        end

        # Standardize the output data

        # Get training points from the EKP iteration number in the second input term  
        N_iter = min(max_iter, length(get_u(ekiobj)) - 1) # number of paired iterations taken from EKP
        min_iter = min(max_iter, max(1, min_iter))
        input_output_pairs = Utilities.get_training_points(ekiobj, min_iter:(N_iter - 1))
        input_output_pairs_test = Utilities.get_training_points(ekiobj, N_iter:(length(get_u(ekiobj)) - 1)) #  "next" iterations
        # Save data
        @save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

        # plot training points in constrained space
        if case == cases[mask[1]]
            gr(dpi = 300, size = (400, 400))
            inputs_unconstrained = get_inputs(input_output_pairs)
            inputs_constrained = transform_unconstrained_to_constrained(priors, inputs_unconstrained)
            p = plot(
                title = "training points",
                xlims = extrema(inputs_constrained[1, :]),
                xaxis = "F",
                yaxis = "A",
                ylims = extrema(inputs_constrained[2, :]),
            )
            scatter!(p, inputs_constrained[1, :], inputs_constrained[2, :], color = :magenta, label = false)
            inputs_test_unconstrained = get_inputs(input_output_pairs_test)
            inputs_test_constrained = transform_unconstrained_to_constrained(priors, inputs_test_unconstrained)

            scatter!(p, inputs_test_constrained[1, :], inputs_test_constrained[2, :], color = :black, label = false)

            vline!(p, [truth_params_constrained[1]], linestyle = :dash, linecolor = :red, label = false)
            hline!(p, [truth_params_constrained[2]], linestyle = :dash, linecolor = :red, label = false)
            savefig(p, joinpath(figure_save_directory, "training_points.pdf"))
            savefig(p, joinpath(figure_save_directory, "training_points.png"))
        end

        # Process data
        retain_var = 0.95
        encoder_schedule = [
            (quartile_scale(), "in"),
            (decorrelate_structure_mat(retain_var = retain_var), "out"),
        ]
        
        emulator = Emulator(
            mlt,
            input_output_pairs;
            output_structure_matrix = Γy,
            encoder_schedule = encoder_schedule,
            obs_noise_cov = Γy,
        )
        optimize_hyperparameters!(emulator)

        # Check how well the Gaussian Process regression predicts on the
        # true parameters
        #if retained_svd_frac==1.0
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
        xlims = extrema(constrained_posterior_samples[1,:])
        ylims = extrema(constrained_posterior_samples[2,:])
        
        # plot with PairPlots


        gr(dpi = 300, size = (300, 300))
        p = cornerplot(permutedims(constrained_posterior_samples, (2, 1)), label = param_names, compact = true, xlims = xlims, ylims=ylims)
        plot!(p.subplots[1], [truth_params_constrained[1]], seriestype = "vline", w = 1.5, c = :steelblue, ls = :dash) # vline on top histogram
        plot!(p.subplots[3], [truth_params_constrained[2]], seriestype = "hline", w = 1.5, c = :steelblue, ls = :dash) # hline on right histogram
        plot!(p.subplots[2], [truth_params_constrained[1]], seriestype = "vline", w = 1.5, c = :steelblue, ls = :dash) # v & h line on scatter.
        plot!(p.subplots[2], [truth_params_constrained[2]], seriestype = "hline", w = 1.5, c = :steelblue, ls = :dash)
        
        figpath = joinpath(figure_save_directory, "posterior_2d-" * case * ".pdf")
        savefig(figpath)
        figpath = joinpath(figure_save_directory, "posterior_2d-" * case * ".png")
        savefig(figpath)


        # Save data
        save(
            joinpath(data_save_directory, "posterior.jld2"),
            "posterior",
            posterior,
            "input_output_pairs",
            input_output_pairs,
            "truth_params",
            truth_params,
        )
    end
end

main()
