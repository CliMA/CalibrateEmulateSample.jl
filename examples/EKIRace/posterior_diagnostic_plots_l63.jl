# Import modules
using Distributions
using LinearAlgebra
using Random
using JLD2
using Statistics
using Plots
using Plots.Measures

# CES
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("Lorenz63.jl")
include("experiment_config.jl")

n_samples_pushforward = 1000

########################################################################
################## file-certainty for loading ##########################
########################################################################

cfg       = experiment_config(:l63)
method    = method_cases[1]
calib_dir = calib_directory(method, cfg)
N_enss    = cfg.N_ens_sizes
rng_idxs  = collect(1:cfg.n_repeats)

prelim_dir  = joinpath(@__DIR__, "output")
if !isdir(prelim_dir)
    mkdir(prelim_dir)
end
prelim_file = joinpath(prelim_dir, "l63_computed_preliminaries.jld2")
if isfile(prelim_file)
    loaded_data            = JLD2.load(prelim_file)
    x0                     = loaded_data["x0"]
    nx                     = length(x0)
    y                      = loaded_data["y"]
    ic_cov_sqrt            = loaded_data["ic_cov_sqrt"]
    R                      = loaded_data["R"]
    lorenz_config_settings = loaded_data["lorenz_config_settings"]
    observation_config     = loaded_data["observation_config"]

    @info "loaded precomputed preliminary quantities from $(prelim_file)"
else
    throw(ErrorException("preliminaries file not found. \n First run: \n > julia --project calibrate_l63.jl"))
end

homedir             = joinpath(pwd())
data_save_directory = joinpath(homedir, "output", calib_dir)

valid_file_items = []
valid_files      = []
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
    error("No valid posterior files found in $(data_save_directory). Run emulate_sample_l63.jl first.")
end

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

for (N_ens, rng_idx) in valid_file_items
    post_fn = posterior_filename(cfg, N_ens, rng_idx)
    @info "loading case $(post_fn)"
    loaded_p = JLD2.load(joinpath(data_save_directory, post_fn))

    posteriors_by_k          = loaded_p["posteriors_by_k"]
    priors                   = loaded_p["priors"]
    k_values                 = loaded_p["k_values"]
    truth_params             = loaded_p["truth_params"]
    truth_params_constrained = loaded_p["truth_params_constrained"]

    n_par = length(truth_params_constrained)
    ny    = length(y)

    # --- sample from prior for grey background ---
    prior_samples_unconstrained = sample(priors, n_samples_pushforward)
    prior_samples_constrained   = transform_unconstrained_to_constrained(priors, prior_samples_unconstrained)
    prior_param_diffs = reduce(hcat, [prior_samples_constrained[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])
    G_prior = hcat(
        [
            lorenz_forward(
                EnsembleMemberConfig(prior_samples_constrained[:, j]),
                x0 .+ ic_cov_sqrt * rand(Normal(0.0, 1.0), nx, 1),
                lorenz_config_settings,
                observation_config,
            ) for j in 1:n_samples_pushforward
        ]...,
    )

    if !haskey(loaded_p, "pushforward_output_samples")
        error("Pushforward data not found in $(post_fn). Run pushforward_from_posterior_l63.jl first.")
    end
    pf_output   = loaded_p["pushforward_output_samples"]   # (n_samples, n_output, n_k)
    pf_k_values = loaded_p["pushforward_k_values"]

    for k in k_values
        post_dist = posteriors_by_k[k]

        # sample posterior for parameter-space analysis (no forward map needed)
        push_ensemble             = sample(post_dist, n_samples_pushforward)
        constrained_push_ensemble = transform_unconstrained_to_constrained(post_dist, push_ensemble)

        # load precomputed posterior pushforward (produced by pushforward_from_posterior_l63.jl)
        ki    = findfirst(==(k), pf_k_values)
        G_ens = pf_output[:, :, ki]'    # (n_output, n_samples)

        param_diffs = reduce(hcat, [constrained_push_ensemble[:, j] - truth_params_constrained for j in 1:n_samples_pushforward])

        param_mah,  param_lp  = pushforward_metrics(push_ensemble, truth_params)
        output_mah, output_lp = pushforward_metrics(G_ens, y)
        lowbd_par, upbd_par = round.(quantile(Chisq(n_par), [0.01, 0.99]), digits=2)
        lowbd_out, upbd_out = round.(quantile(Chisq(ny),    [0.01, 0.99]), digits=2)
        @info "--- Posterior metrics (N_ens=$(N_ens), rng=$(rng_idx), k=$(k)) ---"
        @info "  param  [d=$(n_par)]: target: [$(lowbd_par), $(upbd_par)]  mahal=$(round(param_mah, digits=2))  -2*logpdf_ratio=$(round(-2*param_lp, digits=2))"
        @info "  output [d=$(ny)]:   target: [$(lowbd_out), $(upbd_out)]  mahal=$(round(output_mah, digits=2))  -2*logpdf_ratio=$(round(-2*output_lp, digits=2))"

        gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)

        # Panel (i): parameters (differences from truth)
        p2 = plot(
            collect(1:n_par),
            zeros(n_par),
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Parameter index",
            ylabel = "parameters (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )

        # Panel (ii): output
        p4 = plot(
            1:ny,
            y,
            ribbon = sqrt.(diag(R)),
            label = "data",
            color = :black,
            linewidth = 4,
            xlabel = "Output index",
            ylabel = "State statistics (output)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )

        # Add prior samples (grey)
        plot!(p2, collect(1:n_par), prior_param_diffs[:, 1],    label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
        plot!(p2, collect(1:n_par), prior_param_diffs[:, 2:end], label = "",              color = :grey, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_prior[:, 1],    label = "prior samples", color = :grey, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_prior[:, 2:end], label = "",             color = :grey, linewidth = 4, linealpha = 0.1)

        # Add posterior samples (green)
        plot!(p2, collect(1:n_par), param_diffs[:, 1],    label = "posterior samples", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p2, collect(1:n_par), param_diffs[:, 2:end], label = "",                  color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_ens[:, 1],    label = "pushforward outputs", color = :lightgreen, linewidth = 4, linealpha = 0.1)
        plot!(p4, 1:ny, G_ens[:, 2:end], label = "",                    color = :lightgreen, linewidth = 4, linealpha = 0.1)

        l = @layout [a b]
        plt = plot(p2, p4, layout = l)

        suffix    = case_suffix(cfg, N_ens, rng_idx)
        figure_fn = "pushforward_from_posterior_$(suffix)_k$(k)_full_ens.png"
        savefig(plt, joinpath(data_save_directory, figure_fn))
        savefig(plt, joinpath(data_save_directory, replace(figure_fn, ".png" => ".pdf")))

        # --- posterior ribbons plot ---
        post_param_quantiles  = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(param_diffs)])'
        post_G_quantiles      = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(G_ens)])'
        prior_param_quantiles = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(prior_param_diffs)])'
        prior_G_quantiles     = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(G_prior)])'

        gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)

        # Panel (i): parameter ribbons
        pa = plot(
            collect(1:n_par),
            zeros(n_par),
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Parameter index",
            ylabel = "parameters (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
        plot!(pa, collect(1:n_par), prior_param_quantiles[:, 2],
              color = :grey, label = "prior", linewidth = 4,
              ribbon = [prior_param_quantiles[:, 2] - prior_param_quantiles[:, 1]  prior_param_quantiles[:, 3] - prior_param_quantiles[:, 2]],
              fillalpha = 0.2)
        plot!(pa, collect(1:n_par), post_param_quantiles[:, 2],
              color = :blue, label = "posterior", linewidth = 4,
              ribbon = [post_param_quantiles[:, 2] - post_param_quantiles[:, 1]  post_param_quantiles[:, 3] - post_param_quantiles[:, 2]],
              fillalpha = 0.2)

        # Panel (ii): output ribbons
        pc = plot(
            1:ny,
            y,
            ribbon = sqrt.(diag(R)),
            label = "data",
            color = :black,
            linewidth = 4,
            xlabel = "Output index",
            ylabel = "State statistics (output)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )
        plot!(pc, 1:ny, prior_G_quantiles[:, 2],
              color = :grey, label = "prior", linewidth = 4,
              ribbon = [prior_G_quantiles[:, 2] - prior_G_quantiles[:, 1]  prior_G_quantiles[:, 3] - prior_G_quantiles[:, 2]],
              fillalpha = 0.2)
        plot!(pc, 1:ny, post_G_quantiles[:, 2],
              color = :blue, label = "posterior", linewidth = 4,
              ribbon = [post_G_quantiles[:, 2] - post_G_quantiles[:, 1]  post_G_quantiles[:, 3] - post_G_quantiles[:, 2]],
              fillalpha = 0.2)

        ribbons_plt = plot(pa, pc, layout = @layout [a b])
        ribbons_fn  = "posterior_ribbons_$(suffix)_k$(k)"
        savefig(ribbons_plt, joinpath(data_save_directory, ribbons_fn * ".png"))
        savefig(ribbons_plt, joinpath(data_save_directory, ribbons_fn * ".pdf"))
    end
end
