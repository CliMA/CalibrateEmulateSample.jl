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

function pushforward_metrics(samples::AbstractMatrix, truth::AbstractVector)
    m = vec(mean(samples, dims=2))
    C_raw = cov(samples, dims=2)
    num_samples = size(samples, 2)
    dim_samples = size(samples, 1)
    r = rank(C_raw)
    if r == num_samples - 1 && r < dim_samples - 1
        @warn "Covariance rank $(r) = num_samples-1 = $(num_samples-1) < dim-1 = $(dim_samples-1). Metric may be inaccurate due to insufficient samples; recommend num_samples > $(dim_samples)."
    end
    λ = max(1e-10, 1e-4 * mean(diag(C_raw)))
    C = Symmetric(C_raw + λ * I)
    dist = MvNormal(m, C)
    pmode = samples[:, argmax(logpdf(dist, samples))]
    diff = m - truth
    mah = diff' * (C \ diff)
    lp  = logpdf(dist, truth) - logpdf(dist, pmode)
    return mah, lp
end

function ensemble_from_posterior_one(cfg, N_ens, rng_idx; method = method_cases[1])
    n_samples_pushforward = 1000

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
    truth_params    = loaded_p["truth_params"]

    (truth_params_obj, _) = loaded_c["truth_params_structure"]

    (truth_params_constrained, structure, sample_range) = if force_case == "const-force"
        ([truth_params_obj.val], nothing, nothing)
    elseif force_case == "vec-force"
        (truth_params_obj.val, nothing, nothing)
    elseif force_case == "flux-force"
        tp, _ = destructure(truth_params_obj.model)
        (tp, truth_params_obj.model, truth_params_obj.sample_range)
    end

    is_const = force_case == "const-force"

    truth_emc     = build_forcing(truth_params_obj, truth_params_constrained, structure, sample_range)
    truth_forcing = forcing(truth_emc, x0)
    xaxis_forcing = isnothing(sample_range) ? range(0, length(truth_forcing) - 1, step = 1) : sample_range

    # --- sample from prior for grey background ---
    prior_samples_unconstrained = sample(priors, n_samples_pushforward)
    prior_samples_constrained   = transform_unconstrained_to_constrained(priors, prior_samples_unconstrained)
    prior_param_diffs = reduce(hcat, [prior_samples_constrained[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])
    prior_forcings    = hcat([forcing(build_forcing(truth_params_obj, prior_samples_constrained[:, j], structure, sample_range), x0) for j in 1:n_samples_pushforward]...)
    G_prior = hcat(
        [
            lorenz_forward(
                build_forcing(truth_params_obj, prior_samples_constrained[:, j], structure, sample_range),
                x0 .+ ic_cov_sqrt * rand(Normal(0.0, 1.0), nx, 1),
                lorenz_config_settings,
                observation_config,
            ) for j in 1:n_samples_pushforward
        ]...,
    )

    if !haskey(loaded_p, "pushforward_output_samples")
        error("Pushforward data not found in $(post_fn). Run push_forward_from_posterior.sbatch first.")
    end
    pf_output   = loaded_p["pushforward_output_samples"]   # (n_samples, n_output, n_k_pos)
    pf_forcing  = loaded_p["pushforward_forcing_samples"]  # (n_samples, n_forcing, n_k_pos)
    pf_k_values = loaded_p["pushforward_k_values"]

    for k in k_values
        post_dist = posteriors_by_k[k]

        # sample posterior for parameter-space analysis (no forward map needed)
        push_ensemble             = sample(post_dist, n_samples_pushforward)
        constrained_push_ensemble = transform_unconstrained_to_constrained(post_dist, push_ensemble)

        # load precomputed posterior pushforward (produced by pushforward_from_posterior_l96.jl)
        ki            = findfirst(==(k), pf_k_values)
        G_ens         = pf_output[:, :, ki]'    # (n_output,  n_samples)
        push_forcings = pf_forcing[:, :, ki]'   # (n_forcing, n_samples)

        ny    = length(y)
        n_par = length(truth_params_constrained)
        param_diffs = reduce(hcat, [constrained_push_ensemble[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])

        # Mahalanobis and logpdf metrics in parameter, forcing, and output spaces
        frc_samples_m = is_const ? push_forcings[1:1, :] : push_forcings
        truth_frc_m   = is_const ? truth_forcing[1:1]    : truth_forcing
        param_mah,   param_lp   = pushforward_metrics(push_ensemble, truth_params)
        forcing_mah, forcing_lp = pushforward_metrics(frc_samples_m, truth_frc_m)
        output_mah,  output_lp  = pushforward_metrics(G_ens, y)
        n_frc = length(truth_frc_m)
        lowbd_par, upbd_par = round.(quantile(Chisq(n_par), [0.01, 0.99]), digits=2)
        lowbd_frc, upbd_frc = round.(quantile(Chisq(n_frc), [0.01, 0.99]), digits=2)
        lowbd_out, upbd_out = round.(quantile(Chisq(ny),    [0.01, 0.99]), digits=2)
        @info "--- Posterior metrics (N_ens=$(N_ens), rng=$(rng_idx), k=$(k)) ---"
        @info "  param   [d=$(n_par)]: target: [$(lowbd_par), $(upbd_par)]  mahal=$(round(param_mah, digits=2))  -2*logpdf_ratio=$(round(-2*param_lp, digits=2))"
        @info "  forcing [d=$(n_frc)]: target: [$(lowbd_frc), $(upbd_frc)]  mahal=$(round(forcing_mah, digits=2))  -2*logpdf_ratio=$(round(-2*forcing_lp, digits=2))"
        @info "  output  [d=$(ny)]: target: [$(lowbd_out), $(upbd_out)]  mahal=$(round(output_mah, digits=2))  -2*logpdf_ratio=$(round(-2*output_lp, digits=2))"

        gr(size = (3 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)

        # Panel (i): parameters
        p2 = if is_const
            hline(
                [0.0],
                label = "solution",
                color = :black,
                linewidth = 4,
                xlabel = "Spatial index",
                ylabel = "parameters (input)",
                left_margin = 15mm,
                bottom_margin = 15mm,
            )
        else
            plot(
                collect(1:n_par),
                zeros(n_par),
                label = "solution",
                color = :black,
                linewidth = 4,
                xlabel = "Spatial index",
                ylabel = "parameters (input)",
                left_margin = 15mm,
                bottom_margin = 15mm,
            )
        end

        # Panel (ii): forcing
        p3 = if is_const
            hline(
                [truth_forcing[1]],
                label = "solution",
                color = :black,
                linewidth = 4,
                xlabel = "Spatial index",
                ylabel = "Forcing (transformed-input)",
                left_margin = 15mm,
                bottom_margin = 15mm,
            )
        else
            plot(
                xaxis_forcing,
                truth_forcing,
                label = "solution",
                color = :black,
                linewidth = 4,
                xlabel = "Spatial index",
                ylabel = "Forcing (transformed-input)",
                left_margin = 15mm,
                bottom_margin = 15mm,
            )
        end

        # Panel (iii): output
        p4 = plot(
            1:ny,
            y,
            ribbon = sqrt.(diag(R)),
            label = "data",
            color = :black,
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "State mean/std (output)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )

        # Add prior samples (grey)
        if is_const
            hline!(p2, [prior_param_diffs[1, 1]], label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
            hline!(p2, prior_param_diffs[1, 2:end], label = "", color = :grey, linewidth = 4, linealpha = 0.1)
            hline!(p3, [prior_forcings[1, 1]], label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
            hline!(p3, prior_forcings[1, 2:end], label = "", color = :grey, linewidth = 4, linealpha = 0.1)
        else
            plot!(p2, collect(1:n_par), prior_param_diffs[:, 1],    label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
            plot!(p2, collect(1:n_par), prior_param_diffs[:, 2:end], label = "",              color = :grey, linewidth = 4, linealpha = 0.1)
            plot!(p3, xaxis_forcing, prior_forcings[:, 1],    label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
            plot!(p3, xaxis_forcing, prior_forcings[:, 2:end], label = "",             color = :grey, linewidth = 4, linealpha = 0.1)
        end
        plot!(p4, 1:ny, G_prior[:, 1],    label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_prior[:, 2:end], label = "",             color = :grey, linewidth = 4, linealpha = 0.1)

        # Add posterior samples (green)
        if is_const
            hline!(p2, [param_diffs[1, 1]], label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
            hline!(p2, param_diffs[1, 2:end], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
            hline!(p3, [push_forcings[1, 1]], label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
            hline!(p3, push_forcings[1, 2:end], label = "", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        else
            plot!(p2, collect(1:n_par), param_diffs[:, 1],    label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
            plot!(p2, collect(1:n_par), param_diffs[:, 2:end], label = "",                  color = :lightgreen, linewidth = 4, linealpha = 0.1)
            plot!(p3, xaxis_forcing, push_forcings[:, 1],    label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
            plot!(p3, xaxis_forcing, push_forcings[:, 2:end], label = "",                  color = :lightgreen, linewidth = 4, linealpha = 0.1)
        end
        plot!(p4, 1:ny, G_ens[:, 1],    label = "pushforward outputs", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_ens[:, 2:end], label = "",                    color = :lightgreen, linewidth = 4, linealpha = 0.1)

        l = @layout [a b c]
        plt = plot(p2, p3, p4, layout = l)

        suffix    = case_suffix(cfg, N_ens, rng_idx)
        figure_fn = "pushforward_from_posterior_$(suffix)_k$(k)_full_ens.png"
        savefig(plt, joinpath(data_save_directory, figure_fn))
        savefig(plt, joinpath(data_save_directory, replace(figure_fn, ".png" => ".pdf")))

        # --- posterior ribbons plot ---
        # Ribbons computed from the pushforward ensembles already in hand.
        post_param_quantiles    = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(param_diffs)])'
        post_forcing_quantiles  = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(push_forcings)])'
        post_G_quantiles        = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(G_ens)])'
        prior_param_quantiles   = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(prior_param_diffs)])'
        prior_forcing_quantiles = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(prior_forcings)])'
        prior_G_quantiles       = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(G_prior)])'

        gr(size = (3 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)

        # Panel (i): parameter ribbons
        pa = if is_const
            hline([0.0], label = "solution", color = :black, linewidth = 4,
                  xlabel = "Spatial index", ylabel = "parameters (input)",
                  left_margin = 15mm, bottom_margin = 15mm)
        else
            plot(collect(1:n_par), zeros(n_par), label = "solution", color = :black, linewidth = 4,
                 xlabel = "Spatial index", ylabel = "parameters (input)",
                 left_margin = 15mm, bottom_margin = 15mm)
        end

        if is_const
            hline!(pa, [prior_param_quantiles[1, 2]], color = :grey, label = "prior median", linewidth = 4)
            hline!(pa, [prior_param_quantiles[1, 1]], color = :grey, label = "prior 5/95%", linewidth = 2, linestyle = :dash)
            hline!(pa, [prior_param_quantiles[1, 3]], color = :grey, label = "", linewidth = 2, linestyle = :dash)
            hline!(pa, [post_param_quantiles[1, 2]], color = :blue, label = "posterior median", linewidth = 4)
            hline!(pa, [post_param_quantiles[1, 1]], color = :blue, label = "posterior 5/95%", linewidth = 2, linestyle = :dash)
            hline!(pa, [post_param_quantiles[1, 3]], color = :blue, label = "", linewidth = 2, linestyle = :dash)
        else
            plot!(pa, collect(1:n_par), prior_param_quantiles[:, 2],
                  color = :grey, label = "prior", linewidth = 4,
                  ribbon = [prior_param_quantiles[:, 2] - prior_param_quantiles[:, 1]  prior_param_quantiles[:, 3] - prior_param_quantiles[:, 2]],
                  fillalpha = 0.2)
            plot!(pa, collect(1:n_par), post_param_quantiles[:, 2],
                  color = :blue, label = "posterior", linewidth = 4,
                  ribbon = [post_param_quantiles[:, 2] - post_param_quantiles[:, 1]  post_param_quantiles[:, 3] - post_param_quantiles[:, 2]],
                  fillalpha = 0.2)
        end

        # Panel (ii): forcing ribbons
        pb = if is_const
            hline([truth_forcing[1]], label = "solution", color = :black, linewidth = 4,
                  xlabel = "Spatial index", ylabel = "Forcing (transformed-input)",
                  ylims = (0, 16), left_margin = 15mm, bottom_margin = 15mm)
        else
            plot(xaxis_forcing, truth_forcing, label = "solution", color = :black, linewidth = 4,
                 xlabel = "Spatial index", ylabel = "Forcing (transformed-input)",
                 ylims = (0, 16), left_margin = 15mm, bottom_margin = 15mm)
        end

        if is_const
            hline!(pb, [prior_forcing_quantiles[1, 2]], color = :grey, label = "prior median", linewidth = 4)
            hline!(pb, [prior_forcing_quantiles[1, 1]], color = :grey, label = "prior 5/95%", linewidth = 2, linestyle = :dash)
            hline!(pb, [prior_forcing_quantiles[1, 3]], color = :grey, label = "", linewidth = 2, linestyle = :dash)
            hline!(pb, [post_forcing_quantiles[1, 2]], color = :blue, label = "posterior median", linewidth = 4)
            hline!(pb, [post_forcing_quantiles[1, 1]], color = :blue, label = "posterior 5/95%", linewidth = 2, linestyle = :dash)
            hline!(pb, [post_forcing_quantiles[1, 3]], color = :blue, label = "", linewidth = 2, linestyle = :dash)
        else
            plot!(pb, xaxis_forcing, prior_forcing_quantiles[:, 2],
                  color = :grey, label = "prior", linewidth = 4,
                  ribbon = [prior_forcing_quantiles[:, 2] - prior_forcing_quantiles[:, 1]  prior_forcing_quantiles[:, 3] - prior_forcing_quantiles[:, 2]],
                  fillalpha = 0.2)
            plot!(pb, xaxis_forcing, post_forcing_quantiles[:, 2],
                  color = :blue, label = "posterior", linewidth = 4,
                  ribbon = [post_forcing_quantiles[:, 2] - post_forcing_quantiles[:, 1]  post_forcing_quantiles[:, 3] - post_forcing_quantiles[:, 2]],
                  fillalpha = 0.2)
        end

        # Panel (iii): output ribbons
        pc = plot(1:ny, y, ribbon = sqrt.(diag(R)), label = "data", color = :black, linewidth = 4,
                  xlabel = "Spatial index", ylabel = "State mean/std (output)",
                  left_margin = 15mm, bottom_margin = 15mm)
        plot!(pc, 1:ny, prior_G_quantiles[:, 2],
              color = :grey, label = "prior", linewidth = 4,
              ribbon = [prior_G_quantiles[:, 2] - prior_G_quantiles[:, 1]  prior_G_quantiles[:, 3] - prior_G_quantiles[:, 2]],
              fillalpha = 0.2)
        plot!(pc, 1:ny, post_G_quantiles[:, 2],
              color = :blue, label = "posterior", linewidth = 4,
              ribbon = [post_G_quantiles[:, 2] - post_G_quantiles[:, 1]  post_G_quantiles[:, 3] - post_G_quantiles[:, 2]],
              fillalpha = 0.2)

        ribbons_plt = plot(pa, pb, pc, layout = @layout [a b c])
        ribbons_fn  = "posterior_ribbons_$(suffix)_k$(k)"
        savefig(ribbons_plt, joinpath(data_save_directory, ribbons_fn * ".png"))
        savefig(ribbons_plt, joinpath(data_save_directory, ribbons_fn * ".pdf"))
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
