#includef(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Random
using JLD2
using NCDatasets
using StatsBase
using Dates

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.Utilities

rng_seed = 42424242
Random.seed!(rng_seed)

include("rebuild_priors.jl")


function main()

    # 2-parameter calibration exp
    #exp_name = "ent-det-calibration"

    # 5-parameter calibration exp
    exp_name = "ent-det-tked-tkee-stab-calibration"

    cases = [
        "GP", # diagonalize, train scalar GP, assume diag inputs
        "RF-prior",
        "RF-nonsep",
        "RF-lr-lr", #Bad kernel for comparison
    ]
    rank_cases = [
        [0], # not used
        [10], # some rank > 0
        1:10, # 
        1:5, # input rank always 5, output rank 1:5
    ]

    case_id = 3
    case = cases[case_id]
    rank_test = rank_cases[case_id]
    n_repeats = 1
    n_iteration = 10



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
        @info "loading data from NC dataset"
        data_set = NCDataset(data_filepath)
        y_truth = Array(data_set.group["reference"]["y_full"]) #ndata
        truth_cov = Array(data_set.group["reference"]["Gamma_full"]) #ndata x ndata
        # Option (i) get data from NCDataset else get from jld2 files.
        output_mat = Array(data_set.group["particle_diags"]["g_full"]) #nens x ndata x nit
        input_mat = Array(data_set.group["particle_diags"]["u"]) #nens x nparam x nit
        input_constrained_mat = Array(data_set.group["particle_diags"]["phi"]) #nens x nparam x nit        

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

        # split out a training set:
        n_test = Int(floor(0.2 * length(good_particles)))
        n_train = Int(ceil(0.8 * length(good_particles)))

        rng_seed = 14225
        rng = Random.MersenneTwister(rng_seed)

        # random train-set
        #        train_idx = shuffle(rng, collect(1:n_test+n_train))[1:n_train]
        # final train earlier iteration, test final iteration
        train_idx = collect(1:(n_test + n_train))[1:n_train]
        test_idx = setdiff(collect(1:(n_test + n_train)), train_idx)

        train_inputs = inputs[:, train_idx]
        train_outputs = outputs[:, train_idx]

        test_inputs = inputs[:, test_idx]
        test_outputs = outputs[:, test_idx]

        train_pairs = DataContainers.PairedDataContainer(train_inputs, train_outputs)
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


    @info "Completed data loading stage"
    println(" ")
    ##############################################
    # [3. ] Build Emulator from calibration data #
    ##############################################

    opt_diagnostics = zeros(length(rank_test), n_repeats, n_iteration)
    train_err = zeros(length(rank_test), n_repeats)
    test_err = zeros(length(rank_test), n_repeats)
    ttt = zeros(length(rank_test), n_repeats)

    @info "Begin Emulation stage"
    # Create GP object
    train_frac = 0.9
    n_cross_val_sets = 2
    max_feature_size = size(get_outputs(train_pairs), 2) * size(get_outputs(train_pairs), 1) * (1 - train_frac)
    for (rank_id, rank_val) in enumerate(rank_test) #test over different ranks for svd-nonsep
        rng_seed = 99330
        rng = Random.MersenneTwister(rng_seed)
        for rep_idx in 1:n_repeats
            @info "Test rank: $(rank_val) from $(collect(rank_test))"
            @info "Repeat: $(rep_idx)"

            overrides = Dict(
                "verbose" => true,
                "train_fraction" => train_frac,
                "scheduler" => DataMisfitController(terminate_at = 1000),
                "cov_sample_multiplier" => 0.2,
                "n_iteration" => n_iteration,
                "n_features_opt" => Int(floor((max_feature_size / 5))),# here: /5 with rank <= 3 works
                "localization" => SEC(0.05),
                "n_ensemble" => 100,
                "n_cross_val_sets" => n_cross_val_sets,
            )
            if case == "RF-prior"
                overrides = Dict("verbose" => true, "cov_sample_multiplier" => 0.01, "n_iteration" => 0)
            end
            nugget = 1e-6 # 1e-6
            input_dim = size(get_inputs(train_pairs), 1)
            output_dim = size(get_outputs(train_pairs), 1)
            decorrelate = true

            if case == "GP"
                rank_val = 0
                gppackage = Emulators.SKLJL()
                pred_type = Emulators.YType()
                mlt = GaussianProcess(
                    gppackage;
                    kernel = nothing, # use default squared exponential kernel
                    prediction_type = pred_type,
                    noise_learn = false,
                )
            elseif case ∈ ["RF-lr-lr"]
                kernel_structure = SeparableKernel(LowRankFactor(5, nugget), LowRankFactor(rank_val, nugget))
                n_features = 500

                mlt = VectorRandomFeatureInterface(
                    n_features,
                    input_dim,
                    output_dim,
                    rng = rng,
                    kernel_structure = kernel_structure,
                    optimizer_options = overrides,
                )

            elseif case ∈ ["RF-nonsep", "RF-prior"]
                kernel_structure = NonseparableKernel(LowRankFactor(rank_val, nugget))
                n_features = 500

                mlt = VectorRandomFeatureInterface(
                    n_features,
                    input_dim,
                    output_dim,
                    rng = rng,
                    kernel_structure = kernel_structure,
                    optimizer_options = overrides,
                )
            end
            
            # data processing:
            retain_var = 0.9
            encoder_schedule = (decorrelate_structure_mat(retain_var=retain_var), "in_and_out")
            
            # Fit an emulator to the data
            ttt[rank_id, rep_idx] = @elapsed begin
                emulator = Emulator(
                    mlt,
                    train_pairs;
                    input_structure_matrix = cov(prior),
                    output_structure_matrix = truth_cov,
                    encoder_schedule=encoder_schedule,
                )

                # Optimize the GP hyperparameters for better fit
                optimize_hyperparameters!(emulator)
            end

            # error of emulator on the training points: (no denoised training data)
            train_err_tmp = [0.0]
            for i in 1:size(train_inputs, 2)
                train_mean, _ = Emulators.predict(emulator, train_inputs[:, i:i], transform_to_real = true) # 3x1
                train_err_tmp[1] += norm(train_mean - train_outputs[:, i])
            end
            train_err[rank_id, rep_idx] = 1 / size(train_inputs, 2) * train_err_tmp[1]

            # error of emulator on test points: 
            test_err_tmp = [0.0]
            for i in 1:size(test_inputs, 2)
                test_mean, _ = Emulators.predict(emulator, test_inputs[:, i:i], transform_to_real = true) # 3x1
                test_err_tmp[1] += norm(test_mean - test_outputs[:, i])
            end
            test_err[rank_id, rep_idx] = 1 / size(test_inputs, 2) * test_err_tmp[1]

            @info "train error ($(size(train_inputs,2)) pts): $(train_err[rank_id,rep_idx])"
            @info "test error ($(size(test_inputs, 2)) pts): $(test_err[rank_id,rep_idx])"
            if !(case ∈ ["GP", "RF-prior"])
                opt_diagnostics[rank_id, rep_idx, :] = get_optimizer(mlt)[1] #length-1 vec of vec -> vec
                eki_conv_filepath = joinpath(data_save_directory, "$(case)_$(rank_val)_eki-conv.jld2")
                save(eki_conv_filepath, "opt_diagnostics", opt_diagnostics)
            end

            #            emulator_filepath = joinpath(data_save_directory, "$(case)_$(rank_val)_emulator.jld2")
            #            save(emulator_filepath, "emulator", emulator)

            JLD2.save(
                joinpath(data_save_directory, case * "_edmf_rank_test_results.jld2"),
                "rank_test",
                collect(rank_test),
                "timings",
                ttt,
                "train_err",
                train_err,
                "test_err",
                test_err,
            )
        end
    end




    @info "Finished Emulation stage"
end

main()
