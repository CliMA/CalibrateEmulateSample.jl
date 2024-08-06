#includef(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))
PLOT_FLAG = false

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
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
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.Utilities

rng_seed = 42424242
Random.seed!(rng_seed)


function main()

    # 2-parameter calibration exp
    #exp_name = "ent-det-calibration"

    # 5-parameter calibration exp
    exp_name = "ent-det-tked-tkee-stab-calibration"

    cases = [
        "GP", # diagonalize, train scalar GP, assume diag inputs
        "RF-prior",
        "RF-vector-svd-nonsep",
    ]
    case = cases[3]

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
                savefig(p, joinpath(figure_save_directory, "$(case)_output_" * string(plot_i) * ".png"))
            end
            println("finished plotting ensembles.")
        end
        input_output_pairs = DataContainers.PairedDataContainer(inputs, outputs)
    end

    # load and create prior distributions
    # code deprecated due to JLD2
    #=   
    prior_filepath = joinpath(exp_dir, "prior.jld2")
    if !isfile(prior_filepath)
        LoadError("prior file \"prior.jld2\" not found in directory \"" * exp_dir * "/\"")
    else
        prior_dict_raw = load(prior_filepath) #using JLD2
        prior = prior_dict_raw["prior"]
    end
    =#
    # build prior if jld2 does not work
    function get_prior_config(s::SS) where {SS <: AbstractString}
        config = Dict()
        if s == "ent-det-calibration"
            config["constraints"] =
                Dict("entrainment_factor" => [bounded(0.0, 1.0)], "detrainment_factor" => [bounded(0.0, 1.0)])
            config["prior_mean"] = Dict("entrainment_factor" => 0.13, "detrainment_factor" => 0.51)
            config["unconstrained_σ"] = 1.0
        elseif s == "ent-det-tked-tkee-stab-calibration"
            config["constraints"] = Dict(
                "entrainment_factor" => [bounded(0.0, 1.0)],
                "detrainment_factor" => [bounded(0.0, 1.0)],
                "tke_ed_coeff" => [bounded(0.01, 1.0)],
                "tke_diss_coeff" => [bounded(0.01, 1.0)],
                "static_stab_coeff" => [bounded(0.01, 1.0)],
            )
            config["prior_mean"] = Dict(
                "entrainment_factor" => 0.13,
                "detrainment_factor" => 0.51,
                "tke_ed_coeff" => 0.14,
                "tke_diss_coeff" => 0.22,
                "static_stab_coeff" => 0.4,
            )
            config["unconstrained_σ"] = 1.0
        else
            throw(ArgumentError("prior for experiment $s not found, please implement in uq_for_edmf.jl"))
        end
        return config
    end
    prior_config = get_prior_config(exp_name)
    means = prior_config["prior_mean"]
    std = prior_config["unconstrained_σ"]
    constraints = prior_config["constraints"]

    prior = combine_distributions([
        ParameterDistribution(
            Dict("name" => name, "distribution" => Parameterized(Normal(mean, std)), "constraint" => constraints[name]),
        ) for (name, mean) in means
    ])

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

    println("Completed calibration loading stage")
    println(" ")
    ##############################################
    # [3. ] Build Emulator from calibration data #
    ##############################################
    println("Begin Emulation stage")
    # Create GP object

    overrides = Dict(
        "verbose" => true,
        "train_fraction" => 0.85,
        "scheduler" => DataMisfitController(terminate_at = 100),
        "cov_sample_multiplier" => 1.0,
        "n_iteration" => 15,
        "n_features_opt" => 200,
        "localization" => SEC(0.05),
    )
    if case == "RF-prior"
        overrides = Dict("verbose" => true, "cov_sample_multiplier" => 0.01, "n_iteration" => 0)
    end
    nugget = 1e-6
    rng_seed = 99330
    rng = Random.MersenneTwister(rng_seed)
    input_dim = size(get_inputs(input_output_pairs), 1)
    output_dim = size(get_outputs(input_output_pairs), 1)
    if case == "GP"
        overrides = Dict(
            "verbose" => true,
            "train_fraction" => 0.9, #95
            "scheduler" => DataMisfitController(terminate_at = 1e5),
            "cov_sample_multiplier" => 0.4,
            "n_features_opt" => 200,
            "n_iteration" => 15,
            #    "n_ensemble" => 20,
            #           "localization" => SEC(1.0, 0.01), # localization / sample error correction for small ensembles
        )
    elseif case ∈ ["RF-vector-svd-nonsep", "RF-prior"]
        kernel_structure = NonseparableKernel(LowRankFactor(1, nugget))
        n_features = 500

        gppackage = Emulators.SKLJL()
        pred_type = Emulators.YType()
        mlt = GaussianProcess(
            gppackage;
            kernel = nothing, # use default squared exponential kernel
            prediction_type = pred_type,
            noise_learn = false,
        )
    elseif case ∈ ["RF-vector-svd-nonsep"]
        kernel_structure = NonseparableKernel(LowRankFactor(3, nugget))
        n_features = 500

        mlt = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            rng = rng,
            kernel_structure = kernel_structure,
            optimizer_options = overrides,
        )
    elseif case ∈ ["RF-vector-nosvd-nonsep"]
        kernel_structure = NonseparableKernel(LowRankFactor(3, nugget))
        n_features = 500

        mlt = VectorRandomFeatureInterface(
            n_features,
            input_dim,
            output_dim,
            rng = rng,
            kernel_structure = kernel_structure,
            optimizer_options = overrides,
        )
        decorrelate = false
    end

    # Fit an emulator to the data
    normalized = true

    emulator = Emulator(
        mlt,
        input_output_pairs;
        obs_noise_cov = truth_cov,
        normalize_inputs = normalized,
        decorrelate = decorrelate,
    )

    # Optimize the GP hyperparameters for better fit
    optimize_hyperparameters!(emulator)
    if case ∈ ["RF-vector-nosvd-nonsep", "RF-vector-svd-nonsep"]
        push!(opt_diagnostics, get_optimizer(mlt)[1]) #length-1 vec of vec -> vec
    end

    # plot eki convergence plot
    if length(opt_diagnostics) > 0
        err_cols = reduce(hcat, opt_diagnostics) #error for each repeat as columns?

        #save data
        error_filepath = joinpath(data_save_directory, "eki_conv_error.jld2")
        save(error_filepath, "error", err_cols)

        # print all repeats
        f5 = Figure(resolution = (1.618 * 300, 300), markersize = 4)
        ax_conv = Axis(f5[1, 1], xlabel = "Iteration", ylabel = "max-normalized error")
        if n_repeats == 1
            lines!(ax_conv, collect(1:size(err_cols, 1))[:], err_cols[:], solid_color = :blue) # If just one repeat
        else
            for idx in 1:size(err_cols, 1)
                err_normalized = (err_cols' ./ err_cols[1, :])' # divide each series by the max, so all errors start at 1
                series!(ax_conv, err_normalized', solid_color = :blue)
            end
        end
        save(joinpath(figure_save_directory, "eki-conv_$(case).png"), f5, px_per_unit = 3)
        save(joinpath(figure_save_directory, "eki-conv_$(case).pdf"), f5, px_per_unit = 3)

    end

    emulator_filepath = joinpath(data_save_directory, "$(case)_emulator.jld2")
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
    new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 5000, discard_initial = 0)

    # Now begin the actual MCMC
    println("Begin MCMC - with step size ", new_step)
    chain = MarkovChainMonteCarlo.sample(mcmc, 300_000; stepsize = new_step, discard_initial = 2_000)
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

    mcmc_filepath = joinpath(data_save_directory, "$(case)_mcmc_and_chain.jld2")
    save(mcmc_filepath, "mcmc", mcmc, "chain", chain)

    posterior_filepath = joinpath(data_save_directory, "$(case)_posterior.jld2")
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
