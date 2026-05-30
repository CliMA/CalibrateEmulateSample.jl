# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Dates

# CES
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

include("Lorenz63.jl")
include("experiment_config.jl")

verbose_flag = false
save_all_ekp = true

########################################################################
############### Deterministic problem setup ############################
########################################################################

function build_setup(cfg)
    N_iter      = cfg.N_iter
    N_ens_sizes = cfg.N_ens_sizes
    terminate_at = cfg.terminate_at
    # Seeded so every array task sees the same rng_seeds list.
    rng_seeds = randperm(MersenneTwister(20260529), 1_000_000)[1:cfg.n_repeats]

    nx = 3
    nu = 2
    truth_params = EnsembleMemberConfig([28.0, 8.0 / 3.0])

    prior_r = constrained_gaussian("rho", exp(3.3), 4.153, 0, Inf)
    prior_b = constrained_gaussian("beta", exp(1.2), 2.016, 0, Inf)
    prior = combine_distributions([prior_r, prior_b])
    T = 40.0

    rng_seed_init = 11
    rng_i = MersenneTwister(rng_seed_init)

    t = 0.01
    T_long = 1000.0
    picking_initial_condition = LorenzConfig(t, T_long)
    x_initial = rand(rng_i, Normal(0.0, 1.0), nx)
    x_spun_up = lorenz_solve(truth_params, x_initial, picking_initial_condition)
    x0 = x_spun_up[:, end]

    ny = 9
    lorenz_config_settings = LorenzConfig(t, T)
    T_start = 30.0
    T_end = T
    observation_config = ObservationConfig(T_start, T_end)
    y = lorenz_forward(truth_params, x0, lorenz_config_settings, observation_config)

    multiple = 36
    window = T_end - T_start
    T_R = multiple * window + T_start
    R_config = LorenzConfig(t, T_R)
    R_run = lorenz_solve(truth_params, x_initial, R_config)
    R_sample_size = Int(ceil(multiple))
    R_samples = zeros(ny, R_sample_size)
    for ii in 1:R_sample_size
        local_obs_config = ObservationConfig(T_start + (ii - 1) * window, T_start + ii * window)
        R_samples[:, ii] = stats(R_run, R_config, local_obs_config)
    end
    R = cov(R_samples, dims = 2)
    R_inv_var = sqrt(inv(R))

    covT = 2000.0
    cov_solve = lorenz_solve(truth_params, x0, LorenzConfig(t, covT))
    ic_cov = 0.1 * cov(cov_solve, dims = 2)
    ic_cov_sqrt = sqrt(ic_cov)

    configuration = Dict(
        "N_iter"       => N_iter,
        "N_ens_sizes"  => N_ens_sizes,
        "terminate_at" => terminate_at,
        "rng_seeds"    => rng_seeds,
    )

    return (;
        nx, nu, ny,
        truth_params, prior,
        x0, x_initial,
        y, R, R_inv_var,
        lorenz_config_settings, observation_config, ic_cov_sqrt,
        rng_seeds, configuration,
        N_iter, terminate_at,
    )
end

# Write prior files for all 4 method dirs, idempotent.
function write_priors(cfg, setup, output_dir)
    for method_name in method_cases
        per_method_dir = joinpath(output_dir, calib_directory(method_name, cfg))
        mkpath(per_method_dir)
        pf = joinpath(per_method_dir, prior_filename(cfg))
        if !isfile(pf)
            pf_tmp = pf * ".tmp.$(getpid())"
            JLD2.save(pf_tmp, "prior", setup.prior)
            try
                mv(pf_tmp, pf)
            catch
                rm(pf_tmp; force = true)
            end
        end
    end
end

########################################################################
############### Per-cell calibration (all 4 methods) ##################
########################################################################

function calibrate_one(cfg, setup, N_ens, rng_idx, output_dir)
    rng = MersenneTwister(setup.rng_seeds[rng_idx])
    @info "Calibrating (N_ens=$(N_ens), rng_idx=$(rng_idx))"

    initial_params = construct_initial_ensemble(rng, setup.prior, N_ens)
    methods = [
        Inversion(),
        TransformInversion(),
        GaussNewtonInversion(setup.prior),
        Unscented(setup.prior),
    ]

    conv_cell   = fill(NaN, 4)
    params_cell = zeros(4, setup.nu)
    output_cell = zeros(4, setup.ny)

    for (kk, method) in enumerate(methods)
        @info "Method: $(nameof(typeof(method)))"
        if isa(method, Unscented)
            ekpobj = EKP.EnsembleKalmanProcess(
                setup.y,
                setup.R,
                deepcopy(method);
                rng = copy(rng),
                verbose = verbose_flag,
                localization_method = NoLocalization(),
                scheduler = DataMisfitController(terminate_at = setup.terminate_at),
            )
        else
            ekpobj = EKP.EnsembleKalmanProcess(
                initial_params,
                setup.y,
                setup.R,
                deepcopy(method);
                rng = copy(rng),
                verbose = verbose_flag,
                localization_method = NoLocalization(),
                scheduler = DataMisfitController(terminate_at = setup.terminate_at),
            )
        end
        Ne = get_N_ens(ekpobj)

        ens_mean_final = zeros(setup.nu)
        G_ens_mean_final = zeros(setup.ny)
        for i in 1:setup.N_iter
            params_i = get_ϕ_final(setup.prior, ekpobj)
            ens_mean = mean(params_i, dims = 2)[:]
            G_ens_mean = lorenz_forward(
                EnsembleMemberConfig(ens_mean),
                setup.x0 .+ setup.ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), setup.nx, 1),
                setup.lorenz_config_settings,
                setup.observation_config,
            )
            RMSE_e = norm(setup.R_inv_var * (setup.y - G_ens_mean[:])) / sqrt(setup.ny)
            @info "RMSE (at G(u_mean)): $(RMSE_e)"
            ens_mean_final = ens_mean
            G_ens_mean_final = G_ens_mean[:]
            G_ens = hcat(
                [
                    lorenz_forward(
                        EnsembleMemberConfig(params_i[:, j]),
                        (setup.x0 .+ setup.ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), setup.nx, Ne))[:, j],
                        setup.lorenz_config_settings,
                        setup.observation_config,
                    ) for j in 1:Ne
                ]...,
            )
            terminated = EKP.update_ensemble!(ekpobj, G_ens)
            if !isnothing(terminated)
                conv_cell[kk] = i * Ne
                break
            end
        end
        params_cell[kk, :] = ens_mean_final
        output_cell[kk, :] = G_ens_mean_final
        if isnan(conv_cell[kk])
            conv_cell[kk] = setup.N_iter * Ne
        end

        if save_all_ekp
            per_method_dir = joinpath(output_dir, calib_directory(nameof(typeof(method)), cfg))
            JLD2.save(
                joinpath(per_method_dir, ekp_filename(cfg, N_ens, rng_idx)),
                "N_ens", N_ens,
                "method", method,
                "ekpobj", ekpobj,
            )
            u_stored = get_u(ekpobj, return_array = false)
            g_stored = get_g(ekpobj, return_array = false)
            JLD2.save(
                joinpath(per_method_dir, results_filename(cfg, N_ens, rng_idx)),
                "y", setup.y,
                "R", setup.R,
                "inputs", u_stored,
                "outputs", g_stored,
                "truth_params_structure", setup.truth_params,
            )
        end
    end

    return (conv_cell, params_cell, output_cell)
end

########################################################################
############### Summary helpers ########################################
########################################################################

function save_combined_summary(cfg, setup, output_dir, conv, pars, outs)
    data_filename = joinpath(output_dir, summary_filename(cfg))
    JLD2.save(
        data_filename,
        "configuration",   setup.configuration,
        "method_names",    method_names,
        "conv_alg_iters",  conv,
        "final_parameters", pars,
        "final_model_output", outs,
    )
    @info "Saved summary to $(data_filename)"
end

########################################################################
############### Main dispatcher ########################################
########################################################################

function main()
    cfg        = experiment_config(:l63)
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)

    setup = build_setup(cfg)
    write_priors(cfg, setup, output_dir)

    tasks = flat_tasks(cfg)
    idx   = task_index_from_args()

    if isnothing(idx)
        # Serial mode: run every cell, assemble and write combined summary.
        n_ens = length(cfg.N_ens_sizes)
        n_rep = cfg.n_repeats
        conv  = fill(NaN, 4, n_ens, n_rep)
        pars  = zeros(4, n_ens, n_rep, setup.nu)
        outs  = zeros(4, n_ens, n_rep, setup.ny)
        for (t, (N_ens, rng_idx)) in enumerate(tasks)
            ee = (t - 1) ÷ n_rep + 1
            rr = (t - 1) % n_rep + 1
            c, p, o = calibrate_one(cfg, setup, N_ens, rng_idx, output_dir)
            conv[:, ee, rr]    = c
            pars[:, ee, rr, :] = p
            outs[:, ee, rr, :] = o
        end
        save_combined_summary(cfg, setup, output_dir, conv, pars, outs)
    else
        # Array mode: run one cell; ekp/results files are written inside calibrate_one.
        if idx < 1 || idx > length(tasks)
            error("SLURM_ARRAY_TASK_ID $(idx) out of range 1:$(length(tasks))")
        end
        N_ens, rng_idx = tasks[idx]
        calibrate_one(cfg, setup, N_ens, rng_idx, output_dir)
    end
end

main()
