#include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))
PLOT_FLAG = true

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Plots
using Random
using JLD2
using NCDatasets
using StatsBase
using Dates

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.Utilities

rng_seed = 42424242
Random.seed!(rng_seed)


function main()

    # 2-parameter calibration exp
    exp_name = "ent-det-calibration"

    # 5-parameter calibration exp
    #exp_name = "ent-det-tked-tkee-stab-calibration"


    # Output figure save directory
    figure_save_directory = joinpath(@__DIR__, "output", exp_name, string(Dates.today()))
    data_save_directory = joinpath(@__DIR__, "output", exp_name, string(Dates.today()))
    if !isdir(figure_save_directory)
        mkpath(figure_save_directory)
    end
    if !isdir(data_save_directory)
        mkpath(data_save_directory)
    end
    println("experiment running in: ", @__DIR__)
    println("data saved in: ", figure_save_directory)
    println("figures saved in: ", data_save_directory)

    #############
    # Pipeline: #
    #############

    # [1. ] Obtain scenario information (here all loaded from file)
    # [a.]  Obtain truth sample(s),
    # [b.]  Obtain observation and model-error noise 
    # [2. ] Obtain calibration information  (here using saved input-output data, from EKI experiements)
    # [a.]  Obtain calibration priors
    # [b.]  Obtain calibration data
    # [3. ] Build Emulator from calibration data
    # [4. ] Run Emulator-based MCMC to obtain joint parameter distribution

    ###################################################
    # [1. & 2. ] Obtain scenario and calibration data #
    ###################################################

    exp_dir = joinpath(@__DIR__, exp_name)
    if !isdir(exp_dir)
        LoadError(
            "experiment data directory \"" *
            exp_dir *
            "/\" not found. Please unzip the compressed file \"" *
            exp_name *
            ".zip\" and retry.",
        )
    end
    data_filepath = joinpath(exp_dir, "Diagnostics.nc")
    if !isfile(data_filepath)
        LoadError("experiment data file \"Diagnostics.nc\" not found in directory \"" * exp_dir * "/\"")
    else
        y_truth = Array(NCDataset(data_filepath).group["reference"]["y_full"]) #ndata
        truth_cov = Array(NCDataset(data_filepath).group["reference"]["Gamma_full"]) #ndata x ndata
        # Option (i) get data from NCDataset else get from jld2 files.
        output_mat = Array(NCDataset(data_filepath).group["particle_diags"]["g_full"]) #nens x ndata x nit
        input_mat = Array(NCDataset(data_filepath).group["particle_diags"]["u"]) #nens x nparam x nit
        input_constrained_mat = Array(NCDataset(data_filepath).group["particle_diags"]["phi"]) #nens x nparam x nit        

        # to be consistent, we wish to go from nens x nparam x nit arrays to nparam x (nit x nens) 
        inputs =
            reshape(permutedims(input_mat, (2, 3, 1)), (size(input_mat, 2), size(input_mat, 3) * size(input_mat, 1)))
        inputs_constrained = reshape(
            permutedims(input_constrained_mat, (2, 3, 1)),
            (size(input_constrained_mat, 2), size(input_constrained_mat, 3) * size(input_constrained_mat, 1)),
        )

        # to be consistent, we wish to go from nens x ndata x nit arrays to ndata x (nit x nens) 
        outputs = reshape(
            permutedims(output_mat, (2, 3, 1)),
            (size(output_mat, 2), size(output_mat, 3) * size(output_mat, 1)),
        )

        #EDMF data often contains rows/columns that are NC uses filled values - these must be removed
        println("preprocessing data...")
        isnan_inf_or_filled(x) = isnan(x) || isinf(x) || x ≈ NCDatasets.fillvalue(typeof(x))
        good_values = .!isnan_inf_or_filled.(outputs)
        good_particles = collect(1:size(outputs, 2))[all(.!isnan_inf_or_filled.(outputs), dims = 1)[:]] # get good columns

        inputs = inputs[:, good_particles]
        inputs_constrained = inputs_constrained[:, good_particles]
        outputs = outputs[:, good_particles]

        good_datadim = collect(1:size(outputs, 1))[all(.!isnan_inf_or_filled.(outputs), dims = 2)[:]] # get good rows
        outputs = outputs[good_datadim, :]
        y_truth = y_truth[good_datadim]
        truth_cov = truth_cov[good_datadim, good_datadim]

        # quick plots of data
        if PLOT_FLAG
            println("plotting ensembles...")
            for plot_i in 1:size(outputs, 1)
                p = scatter(inputs_constrained[1, :], inputs_constrained[2, :], zcolor = outputs[plot_i, :])
                savefig(p, joinpath(figure_save_directory, "output_" * string(plot_i) * ".png"))
            end
            println("finished plotting ensembles.")
        end
        input_output_pairs = DataContainers.PairedDataContainer(inputs, outputs)

    end

    # load and create prior distributions
    prior_filepath = joinpath(exp_dir, "prior.jld2")
    if !isfile(prior_filepath)
        LoadError("prior file \"prior.jld2\" not found in directory \"" * exp_dir * "/\"")
    else
        prior_dict_raw = load(prior_filepath) #using JLD2
        marginal_priors =
            ParameterDistributions.ParameterDistribution.(
                prior_dict_raw["distributions"],
                prior_dict_raw["constraints"],
                prior_dict_raw["u_names"],
            )
        prior = ParameterDistributions.combine_distributions(marginal_priors)
    end

    # Option (ii) load EKP object
    # max_ekp_it = 10 # use highest available iteration file
    # ekp_filename = "ekpobj_iter_"*max_ekp_it*"10.jld2"
    # ekp_filepath = joinpath(exp_dir,ekp_filename) 
    # if !isfile(prior_filepath)
    #     LoadError("ensemble object file \""*ekp_filename*"\" not found in directory \""*exp_dir*"/\"")
    # else
    #     ekpobj = load(ekp_filepath)["ekp"]
    # end
    # input_output_pairs = Utilities.get_training_points(ekpobj, max_ekp_it)

    println("Completed calibration stage")
    println(" ")
    ##############################################
    # [3. ] Build Emulator from calibration data #
    ##############################################
    println("Begin Emulation stage")
    # Create GP object

    gppackage = Emulators.SKLJL()
    pred_type = Emulators.YType()
    gauss_proc = GaussianProcess(
        gppackage;
        kernel = nothing, # use default squared exponential kernel
        prediction_type = pred_type,
        noise_learn = false,
    )

    # Fit an emulator to the data
    normalized = true

    emulator = Emulator(gauss_proc, input_output_pairs; obs_noise_cov = truth_cov, normalize_inputs = normalized)

    # Optimize the GP hyperparameters for better fit
    optimize_hyperparameters!(emulator)

    emulator_filepath = joinpath(data_save_directory, "emulator.jld2")
    save(emulator_filepath, "emulator", emulator)

    println("Finished Emulation stage")
    println(" ")
    ########################################################################
    # [4. ] Run Emulator-based MCMC to obtain joint parameter distribution #
    ########################################################################
    println("Begin Sampling stage")

    u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
    println("initial parameters: ", u0)

    # determine a good step size
    yt_sample = y_truth
    mcmc = MCMCWrapper(RWMHSampling(), yt_sample, prior, emulator; init_params = u0)
    new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

    # Now begin the actual MCMC
    println("Begin MCMC - with step size ", new_step)
    chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

    mcmc_filepath = joinpath(data_save_directory, "mcmc_and_chain.jld2")
    save(mcmc_filepath, "mcmc", mcmc, "chain", chain)

    posterior_filepath = joinpath(data_save_directory, "posterior.jld2")
    save(posterior_filepath, "posterior", posterior)

    println("Finished Sampling stage")
    # extract some statistics
    post_mean = mean(posterior)
    post_cov = cov(posterior)
    println("mean in of posterior samples (taken in comp space)")
    println(post_mean)
    println("covariance of posterior samples (taken in comp space)")
    println(post_cov)
    println("transformed posterior mean from comp to phys space")
    println(transform_unconstrained_to_constrained(posterior, post_mean))

    # map to physical space
    posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
    transformed_posterior_samples =
        mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)
    println("mean of posterior samples (taken in phys space)")
    println(mean(transformed_posterior_samples, dims = 2))
    println("cov of posterior samples (taken in phys space)")
    println(cov(transformed_posterior_samples, dims = 2))
end

main()
