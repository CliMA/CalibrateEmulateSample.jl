# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Flux
using BSON
using Dates
using Plots
using Plots.Measures


# CES
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("Lorenz96.jl") # Contains Lorenz 96 source code
include("experiment_config.jl")

verbose_flag = false
save_all_ekp = true

n_samples_pushforward = 100 # num. samples to pushforward through lorenz


########################################################################
################## file-certainty for loading ##########################
########################################################################

@assert EXPERIMENT in (:l96_const, :l96_vec, :l96_flux) "For calibrate_l96.jl, set EXPERIMENT to :l96_const, :l96_vec, or :l96_flux in experiment_config.jl"
cfg        = experiment_config(EXPERIMENT)
method     = method_cases[1]  # method_cases defined in experiment_config.jl
calib_dir  = calib_directory(method, cfg)
force_case = cfg.force_case
N_enss     = cfg.N_ens_sizes
rng_idxs   = collect(1:cfg.n_repeats)

# get some preliminaries
prelim_dir = joinpath(@__DIR__, "output")
if !isdir(prelim_dir)
    mkdir(prelim_dir)
end
prelim_file = joinpath(prelim_dir, "l96_computed_preliminaries_$(force_case).jld2")
if isfile(prelim_file)
    loaded_data = JLD2.load(prelim_file)
    x0 = loaded_data["x0"]
    nx = length(x0)
    y = loaded_data["y"]
    ic_cov_sqrt = loaded_data["ic_cov_sqrt"]
    R = loaded_data["R"]
    R_inv_var = loaded_data["R_inv_var"]
    lorenz_config_settings = loaded_data["lorenz_config_settings"]
    observation_config = loaded_data["observation_config"]

    @info "loaded precomputed preliminary quantities from $(prelim_file)"
else
    throw(ErrorException("preliminaries files not found. \n First run: \n > julia --project calibrate_l96.jl"))
end
### determine valid files before we try to load

homedir = joinpath(pwd())
data_save_directory = joinpath(homedir, "output", calib_dir)

valid_file_items = []
valid_files = []
for N_ens in N_enss
    for rng_idx in rng_idxs
        data_file = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))
        if isfile(data_file)
            push!(valid_files, case_suffix(cfg, N_ens, rng_idx))
            push!(valid_file_items, (N_ens, rng_idx))
        end
    end
end

@info "Pushing forward posteriors through the forward map from valid files:"
display(valid_files)

if isempty(valid_file_items)
    error("No valid posterior files found in $(data_save_directory). Run emulate_sample_l96.jl first.")
end

### Then load data

# Load first valid file to determine parameter dimension and max k
first_loaded = JLD2.load(joinpath(data_save_directory, posterior_filename(cfg, valid_file_items[1]...)))
n_params = length(vec(mean(first_loaded["posteriors_by_k"][1])))

n_k = maximum(
    maximum(JLD2.load(joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx)))["k_values"])
    for (N_ens, rng_idx) in valid_file_items
)

n_rng = length(rng_idxs)
n_ens = length(N_enss)

function pushforward_metrics(samples::AbstractMatrix, truth::AbstractVector)
    m = vec(mean(samples, dims=2))
    C_raw = cov(samples')
    λ = max(1e-10, 1e-4 * mean(diag(C_raw)))
    C = Symmetric(C_raw + λ * I)
    dist = MvNormal(m, C)
    pmode = samples[:, argmax(logpdf(dist, samples))]
    diff = m - truth
    mah = diff' * (C \ diff)
    lp  = logpdf(dist, truth) - logpdf(dist, pmode)
    return mah, lp
end

for (N_ens, rng_idx) in valid_file_items
    calib_fn = results_filename(cfg, N_ens, rng_idx)
    post_fn = posterior_filename(cfg, N_ens, rng_idx)
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
        truth_params_constrained, _ = destructure(truth_params_obj.model)

        (
            truth_params_constrained,
            truth_params_obj.model,
            truth_params_obj.sample_range
        )
    end
    truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)

    is_const = force_case == "const-force"

    truth_emc     = build_forcing(truth_params_obj, truth_params_constrained, structure, sample_range)
    truth_forcing = forcing(truth_emc, x0)
    xaxis_forcing = isnothing(sample_range) ? range(0, length(truth_forcing) - 1, step = 1) : sample_range

    # --- sample from prior for grey background ---
    prior_samples_unconstrained = sample(priors, n_samples_pushforward)
    prior_samples_constrained   = transform_unconstrained_to_constrained(priors, prior_samples_unconstrained)
    prior_param_diffs   = reduce(hcat, [prior_samples_constrained[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])
    prior_forcings      = hcat([forcing(build_forcing(truth_params_obj, prior_samples_constrained[:, j], structure, sample_range), x0) for j in 1:n_samples_pushforward]...)
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
        push_forcings = hcat([forcing(build_forcing(truth_params_obj, constrained_push_ensemble[:, j], structure, sample_range), x0) for j in 1:n_samples_pushforward]...)
        param_diffs   = reduce(hcat, [constrained_push_ensemble[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])

        # Mahalanobis and logpdf metrics in parameter, forcing, and output spaces
        frc_samples_m = is_const ? push_forcings[1:1, :] : push_forcings
        truth_frc_m   = is_const ? truth_forcing[1:1]    : truth_forcing
        param_mah,   param_lp   = pushforward_metrics(constrained_push_ensemble, truth_params_constrained)
        forcing_mah, forcing_lp = pushforward_metrics(frc_samples_m, truth_frc_m)
        output_mah,  output_lp  = pushforward_metrics(G_ens, y)
        @info "--- Posterior metrics (N_ens=$(N_ens), rng=$(rng_idx), k=$(k)) ---"
        @info "  param   [d=$(n_par)]: mahal=$(round(param_mah, digits=2))  logpdf_ratio=$(round(param_lp, digits=2))"
        @info "  forcing [d=$(length(truth_frc_m))]: mahal=$(round(forcing_mah, digits=2))  logpdf_ratio=$(round(forcing_lp, digits=2))"
        @info "  output  [d=$(ny)]: mahal=$(round(output_mah, digits=2))  logpdf_ratio=$(round(output_lp, digits=2))"

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
        plot!(p4, 1:ny, G_ens[:, 2:end], color = :lightgreen, label = "",                   linewidth = 4, linealpha = 0.1)

        l = @layout [a b c]
        plt = plot(p2, p3, p4, layout = l)

        suffix = case_suffix(cfg, N_ens, rng_idx)
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


