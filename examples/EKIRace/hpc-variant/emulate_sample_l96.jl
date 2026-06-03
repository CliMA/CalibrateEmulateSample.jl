# Import modules
include(joinpath(@__DIR__, "..", "..", "ci", "linkfig.jl"))

using Distributions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using StatsPlots
using Plots
using Random
using JLD2
using Dates

# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

include("Lorenz96.jl")
include("experiment_config.jl")

########################################################################
############### Per-cell emulate–sample ################################
########################################################################

function emulate_sample_one(cfg, N_ens, rng_idx; method = method_cases[1])
    calib_dir = calib_directory(method, cfg)
    calib_filename_suffix = case_suffix(cfg, N_ens, rng_idx)
    @info("Perform emulate-sample for L96: \n method: $(calib_dir) \n experiment: $(calib_filename_suffix)")

    homedir = pwd()
    figure_save_directory = joinpath(homedir, "output/", calib_dir)
    data_save_directory   = joinpath(homedir, "output",  calib_dir)
    loaded_calib_files = [
        joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx)),
        joinpath(data_save_directory, prior_filename(cfg)),
        joinpath(data_save_directory, results_filename(cfg, N_ens, rng_idx)),
    ]
    if !isfile(loaded_calib_files[1])
        throw(ErrorException("data files not found. \n First run: \n > julia --project calibrate_l96.jl"))
    end

    ekpobj = load(loaded_calib_files[1])["ekpobj"]
    priors = load(loaded_calib_files[2])["prior"]
    truth_sample = load(loaded_calib_files[3])["y"]
    (truth_params_obj, truth_params_structure) = load(loaded_calib_files[3])["truth_params_structure"]

    force_case = cfg.force_case
    truth_params_constrained = if force_case == "const-force"
        [truth_params_obj.val]
    elseif force_case == "vec-force"
        truth_params_obj.val
    elseif force_case == "flux-force"
        truth_params_model = truth_params_obj.model
        truth_params_constrained, rebuild = destructure(truth_params_model)
        truth_params_constrained
    end
    truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)
    Γy = get_obs_noise_cov(ekpobj)
    n_params = length(truth_params)

    final_params_constrained = get_ϕ_mean_final(priors, ekpobj)
    encoder_kwargs = encoder_kwargs_from(ekpobj, priors)
    retain_var = cfg.retain_var
    nugget = 1e-6

    iter_end = length(get_u(ekpobj)) - 1
    K = min(iter_end - 1, cfg.max_iter)
    if K < 1
        @error("Not enough EKP iterations for emulate-sample loop. Skipping $(calib_filename_suffix)...")
        return
    end
    @info "Running emulate-sample for k = 1:$(K) training iterations"

    posteriors_by_k       = Dict{Int, Any}()
    iop_by_k              = Dict{Int, Any}()
    encoder_schedule_by_k = Dict{Int, Any}()

    for k in 1:K
        rng = Random.MersenneTwister(44011)
        input_output_pairs = Utilities.get_training_points(ekpobj, 1:k)
        @info "k = $(k): training on $(N_ens * k) samples (iterations 1:$(k))"

        ###  Emulate
        csm = N_ens * k < 200 ? 5.0 : 1.0
        overrides = Dict(
            "scheduler" => DataMisfitController(terminate_at = 1000.0),
            "cov_sample_multiplier" => csm,
            "n_iteration" => 10,
            "n_features_opt" => cfg.n_features_opt,
        )
        mlt = ScalarRandomFeatureInterface(
            cfg.n_features,
            n_params,
            rng = rng,
            kernel_structure = SeparableKernel(DiagonalFactor(nugget), OneDimFactor()),
            optimizer_options = overrides,
        )
        #                    encoder_schedule = [(decorrelate_structure_mat(retain_var = retain_var), "in_and_out")]
        
        encoder_schedule = [
            (decorrelate_structure_mat(), "in_and_out"),
            (likelihood_informed(retain_info=retain_var, iters=1:k),"in"),
            (likelihood_informed(retain_info=retain_var),"out"), # iters=1:1 for output space
        ]
        emulator = Emulator(
            mlt,
            input_output_pairs;
            encoder_schedule = deepcopy(encoder_schedule),
            encoder_kwargs = deepcopy(encoder_kwargs),
        )
        optimize_hyperparameters!(emulator)

        # diagnostics
        y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1))
        diff = truth_sample - vec(y_mean)
        Γypluscov = Symmetric(mean(y_var) + Γy)
        println("k=$(k) | ML weighted-MSE (truth): ", mean(diff' * (Γypluscov \ diff)))

        ###  Sample: MCMC
        u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
        mcmc = MCMCWrapper(pCNMHSampling(), truth_sample, priors, emulator; init_params = u0)
        new_step = optimize_stepsize(mcmc; init_stepsize = 0.2, N = 2000, discard_initial = 0)
        println("k=$(k) | Begin MCMC - step size ", new_step)
        chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)
        posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
        println("k=$(k) | post_mean: ", mean(posterior))
        println("k=$(k) | D util: ", det(inv(cov(posterior))))

        # figure
        pp = plot(priors, c = :gray)
        plot!(pp, posterior)
        for (i, sp) in enumerate(pp.subplots)
            vline!(sp, [truth_params_constrained[i]], lc = :black, lw = 4)
            vline!(sp, [final_params_constrained[i]], lc = :magenta, lw = 4)
        end
        figpath = joinpath(figure_save_directory, "l96_posterior_hist_$(calib_filename_suffix)_k$(k)")
        savefig(figpath * ".png")

        posteriors_by_k[k]       = posterior
        iop_by_k[k]              = input_output_pairs
        encoder_schedule_by_k[k] = get_encoder_schedule(emulator)
    end

    save(
        joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx)),
        "posteriors_by_k",          posteriors_by_k,
        "input_output_pairs_by_k",  iop_by_k,
        "encoder_schedules_by_k",   encoder_schedule_by_k,
        "priors",                   priors,
        "truth_params_constrained", truth_params_constrained,
        "final_params_constrained", final_params_constrained,
        "truth_params",             truth_params,
        "k_values",                 collect(1:K),
    )
end

########################################################################
############### Main dispatcher ########################################
########################################################################

function main()
    exp = l96_experiment()
    @assert(
        exp in (:l96_const, :l96_vec, :l96_flux),
        "EXPERIMENT must be :l96_const, :l96_vec, or :l96_flux (got $exp)",
    )
    cfg   = experiment_config(exp)
    tasks = flat_tasks(cfg)
    idx   = task_index_from_args()

    if isnothing(idx)
        for (N_ens, rng_idx) in tasks
            emulate_sample_one(cfg, N_ens, rng_idx)
        end
    else
        if idx < 1 || idx > length(tasks)
            error("SLURM_ARRAY_TASK_ID $(idx) out of range 1:$(length(tasks))")
        end
        N_ens, rng_idx = tasks[idx]
        emulate_sample_one(cfg, N_ens, rng_idx)
    end
end

main()
