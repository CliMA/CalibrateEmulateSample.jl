# Import modules
using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Flux
using BSON
using Dates
ENV["GKSwstype"] = "100"
using Plots
using Plots.Measures


# CES
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("Lorenz96.jl")
include("experiment_config.jl")

########################################################################
############### Per-cell pushforward ###################################
########################################################################

function ensemble_from_posterior_one(cfg, N_ens, rng_idx; method = method_cases[1])
    n_samples_pushforward = 100

    force_case  = cfg.force_case
    calib_dir   = calib_directory(method, cfg)
    calib_fn    = results_filename(cfg, N_ens, rng_idx)
    post_fn     = posterior_filename(cfg, N_ens, rng_idx)

    # load preliminaries
    prelim_dir  = joinpath(@__DIR__, "output")
    prelim_file = joinpath(prelim_dir, "l96_computed_preliminaries_$(force_case).jld2")
    if !isfile(prelim_file)
        throw(ErrorException("preliminaries files not found. \n First run: \n > julia --project calibrate_l96.jl"))
    end
    loaded_data = JLD2.load(prelim_file)
    x0                    = loaded_data["x0"]
    nx                    = length(x0)
    y                     = loaded_data["y"]
    ic_cov_sqrt           = loaded_data["ic_cov_sqrt"]
    R                     = loaded_data["R"]
    lorenz_config_settings = loaded_data["lorenz_config_settings"]
    observation_config    = loaded_data["observation_config"]

    homedir             = joinpath(pwd())
    data_save_directory = joinpath(homedir, "output", calib_dir)

    if !isfile(joinpath(data_save_directory, post_fn))
        @warn "No posterior file found for $(case_suffix(cfg, N_ens, rng_idx)); skipping."
        return
    end

    @info "loading case $(post_fn)"
    loaded_p = JLD2.load(joinpath(data_save_directory, post_fn))
    loaded_c = JLD2.load(joinpath(data_save_directory, calib_fn))

    posteriors_by_k = loaded_p["posteriors_by_k"]
    priors          = loaded_p["priors"]
    k_values        = loaded_p["k_values"]

    (truth_params_obj, _) = loaded_c["truth_params_structure"]

    (truth_params_constrained, structure, sample_range) = if force_case == "const-force"
        ([truth_params_obj.val], nothing, nothing)
    elseif force_case == "vec-force"
        (truth_params_obj.val, nothing, nothing)
    elseif force_case == "flux-force"
        tp, _ = destructure(truth_params_obj.model)
        (tp, truth_params_obj.model, truth_params_obj.sample_range)
    end

    for k in k_values
        post_dist = posteriors_by_k[k]
        push_ensemble = sample(post_dist, n_samples_pushforward)
        # note: push_ensemble is unconstrained
        constrained_push_ensemble = transform_unconstrained_to_constrained(post_dist, push_ensemble)

        G_ens = hcat(
            [
                lorenz_forward(
                    build_forcing(truth_params_obj, constrained_push_ensemble[:, j], structure, sample_range),
                    x0 .+ ic_cov_sqrt * rand(Normal(0.0, 1.0), nx, 1),
                    lorenz_config_settings,
                    observation_config,
                ) for j in 1:n_samples_pushforward
            ]...,
        )

        ny    = length(y)
        n_par = length(truth_params_constrained)

        truth_emc     = build_forcing(truth_params_obj, truth_params_constrained, structure, sample_range)
        truth_forcing = forcing(truth_emc, x0)
        xaxis_forcing = isnothing(sample_range) ? range(0, length(truth_forcing) - 1, step = 1) : sample_range
        push_forcings = hcat([forcing(build_forcing(truth_params_obj, constrained_push_ensemble[:, j], structure, sample_range), x0) for j in 1:n_samples_pushforward]...)
        param_diffs   = reduce(hcat, [constrained_push_ensemble[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])

        gr(size = (3 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
        p2 = plot(
            collect(1:n_par),
            zeros(n_par),
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "parameters(input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
        p3 = plot(
            xaxis_forcing,
            truth_forcing,
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Forcing (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
        p4 = plot(
            1:ny,
            y,
            ribbon = sqrt.(diag(R)),
            label = "data",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "State mean/std output",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )

        plot!(p2, collect(1:n_par), param_diffs[:, 1],    label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p2, collect(1:n_par), param_diffs[:, 2:end], label = "",               color = :lightgreen, linewidth = 4, linealpha = 0.1)

        plot!(p3, xaxis_forcing, push_forcings[:, 1],    label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p3, xaxis_forcing, push_forcings[:, 2:end], label = "",                 color = :lightgreen, linewidth = 4, linealpha = 0.1)

        plot!(p4, 1:ny, G_ens[:, 1],     label = "pushforward outputs", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_ens[:, 2:end], label = "",                    color = :lightgreen, linewidth = 4, linealpha = 0.1)

        l = @layout [a b c]
        plt = plot(p2, p3, p4, layout = l)

        suffix    = case_suffix(cfg, N_ens, rng_idx)
        figure_fn = "pushforward_from_posterior_$(suffix)_k$(k)_full_ens.png"
        savefig(plt, joinpath(data_save_directory, figure_fn))
        savefig(plt, joinpath(data_save_directory, replace(figure_fn, ".png" => ".pdf")))
    end
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
            ensemble_from_posterior_one(cfg, N_ens, rng_idx)
        end
    else
        if idx < 1 || idx > length(tasks)
            error("SLURM_ARRAY_TASK_ID $(idx) out of range 1:$(length(tasks))")
        end
        N_ens, rng_idx = tasks[idx]
        ensemble_from_posterior_one(cfg, N_ens, rng_idx)
    end
end

main()
