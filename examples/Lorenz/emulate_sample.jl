# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
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
using CalibrateEmulateSample.Observations

function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2)
    return norm_factor
end

function get_standardizing_factors(data::Array{FT, 1}) where {FT}
    # Input: data size: N_data*N_ensembles (splatted)
    # Ensemble median of the data
    norm_factor = median(data)
    return norm_factor
end


function main()

    cases = [
        "GP", # diagonalize, train scalar GP, assume diag inputs
        "RF-scalar-diagin", # diagonalize, train scalar RF, assume diag inputs (most comparable to GP)
        "RF-scalar", # diagonalize, train scalar RF, don't asume diag inputs
        "RF-vector-svd-diag",
        "RF-vector-svd-nondiag",
        "RF-vector-nosvd-diag",
        "RF-vector-nosvd-nondiag",
    ]

    #### CHOOSE YOUR CASE: 
    mask = 2:7
    # One day we can use good heuristics here
    # Currently set so that learnt hyperparameters stays relatively far inside the prior. (use "verbose" => true in optimizer overrides to see this info)
    prior_scalings = [[0.0, 0.0], [1e-2, 0.0], [1e-1, 0.0], [1e-2, 1e-2], [1e-2, 1e-1], [1e-2, 1e-2], [1e-2, 1e-2]]
    for (case, scaling) in zip(cases[mask], prior_scalings[mask])

        #case = cases[7]
        println("case: ", case)
        max_iter = 6 # number of EKP iterations to use data from is at most this
        overrides = Dict(
            "verbose" => true,
            "train_fraction" => 0.8,
            "n_iteration" => 40,
            "scheduler" => DataMisfitController(),
            "prior_in_scale" => scaling[1],
            "prior_out_scale" => scaling[2],
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
        truth_sample = load(data_save_file)["truth_sample"]
        truth_sample_mean = load(data_save_file)["truth_sample_mean"]
        truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
        truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)
        Γy = ekiobj.obs_noise_cov
        println(Γy)

        n_params = length(truth_params) # "input dim"
        output_dim = size(Γy, 1)
        ###
        ###  Emulate: Gaussian Process Regression
        ###

        # Emulate-sample settings
        # choice of machine-learning tool
        if case == "GP"
            gppackage = Emulators.GPJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(
                gppackage;
                kernel = nothing, # use default squared exponential kernel
                prediction_type = pred_type,
                noise_learn = false,
            )
        elseif case ∈ ["RF-scalar", "RF-scalar-diagin"]
            n_features = 300
            diagonalize_input = case == "RF-scalar-diagin" ? true : false
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_params,
                rng = rng,
                diagonalize_input = diagonalize_input,
                optimizer_options = overrides,
            )
        elseif case ∈ ["RF-vector-svd-diag", "RF-vector-nosvd-diag", "RF-vector-svd-nondiag", "RF-vector-nosvd-nondiag"]
            # do we want to assume that the outputs are decorrelated in the machine-learning problem?
            diagonalize_output = case ∈ ["RF-vector-svd-diag", "RF-vector-nosvd-diag"] ? true : false
            n_features = 300

            mlt = VectorRandomFeatureInterface(
                n_features,
                n_params,
                output_dim,
                rng = rng,
                diagonalize_output = diagonalize_output, # assume outputs are decorrelated
                optimizer_options = overrides,
            )
        end

        # Standardize the output data
        # Use median over all data since all data are the same type
        truth_sample_norm = vcat(truth_sample...)
        norm_factor = get_standardizing_factors(truth_sample_norm)
        println(size(norm_factor))
        #norm_factor = vcat(norm_factor...)
        norm_factor = fill(norm_factor, size(truth_sample))
        println("Standardization factors")
        println(norm_factor)

        # Get training points from the EKP iteration number in the second input term  
        N_iter = min(max_iter, length(get_u(ekiobj)) - 1) # number of paired iterations taken from EKP

        input_output_pairs = Utilities.get_training_points(ekiobj, N_iter - 1) # 1:N-1

        input_output_pairs_test = Utilities.get_training_points(ekiobj, N_iter:N_iter) # just "next" iteration used for testing
        # Save data
        @save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

        standardize = false
        retained_svd_frac = 1.0
        normalized = true
        # do we want to use SVD to decorrelate outputs
        decorrelate = case ∈ ["RF-vector-nosvd-diag", "RF-vector-nosvd-nondiag"] ? false : true


        emulator = Emulator(
            mlt,
            input_output_pairs;
            obs_noise_cov = Γy,
            normalize_inputs = normalized,
            standardize_outputs = standardize,
            standardize_outputs_factors = norm_factor,
            retained_svd_frac = retained_svd_frac,
            decorrelate = decorrelate,
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
        println(truth_sample_mean) # same, regardless of norm_factor 
        println(" ML predicted variance")
        println(diag(y_var[1], 0))
        println("ML MSE (truth): ")
        println(mean((truth_sample_mean - vec(y_mean)) .^ 2))
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
        truth_sample = truth_sample
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

        constrained_truth_params = transform_unconstrained_to_constrained(posterior, truth_params)
        param_names = get_name(posterior)

        posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
        constrained_posterior_samples =
            mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

        gr(dpi = 300, size = (300, 300))
        p = cornerplot(permutedims(constrained_posterior_samples, (2, 1)), label = param_names, compact = true)
        plot!(p.subplots[1], [constrained_truth_params[1]], seriestype = "vline", w = 1.5, c = :steelblue, ls = :dash) # vline on top histogram
        plot!(p.subplots[3], [constrained_truth_params[2]], seriestype = "hline", w = 1.5, c = :steelblue, ls = :dash) # hline on right histogram
        plot!(p.subplots[2], [constrained_truth_params[1]], seriestype = "vline", w = 1.5, c = :steelblue, ls = :dash) # v & h line on scatter.
        plot!(p.subplots[2], [constrained_truth_params[2]], seriestype = "hline", w = 1.5, c = :steelblue, ls = :dash)
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
