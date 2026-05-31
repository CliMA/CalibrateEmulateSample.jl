# load packages
# CES

using Random
using JLD2
using Statistics
using LinearAlgebra
using Dates

using Plots
using Plots.Measures

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

include("experiment_config.jl")

#### CHOOSE YOUR CASE:
exp = l96_experiment()
@assert exp in (:l96_const, :l96_vec, :l96_flux) "For plot_l96_forcing.jl, set EXPERIMENT to :l96_const, :l96_vec, or :l96_flux in experiment_config.jl"
cfg        = experiment_config(exp)
method     = method_cases[1]
calib_dir  = calib_directory(method, cfg)
force_cases = [cfg.force_case]
N_enss     = cfg.N_ens_sizes
rng_idxs   = collect(1:cfg.n_repeats)

homedir = pwd()

valid_file_items = []
valid_files = []
for force_case in force_cases
    for N_ens in N_enss
        for rng_idx in rng_idxs
            data_save_directory = joinpath(homedir, "output", calib_dir)
            data_file = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))
            if isfile(data_file)
                push!(valid_files, case_suffix(cfg, N_ens, rng_idx))
                push!(valid_file_items, (force_case, N_ens, rng_idx))
            end
        end
    end
end
@info "Plotting valid files:"
display(valid_files)
println(" ")

for ((force_case, N_ens, rng_idx), calib_filename_suffix) in zip(valid_file_items, valid_files)

    @info("Plotting for L96: \n method: $(calib_dir) \n experiment: $(calib_filename_suffix)")

    # load
    data_save_directory = joinpath(homedir, "output", calib_dir)
    figure_save_directory = data_save_directory
    data_file = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))
    truth_params     = load(data_file)["truth_params_constrained"]
    final_params     = load(data_file)["final_params_constrained"]
    posteriors_by_k  = load(data_file)["posteriors_by_k"]
    k_values         = load(data_file)["k_values"]
    prior            = load(data_file)["priors"]

    ekp_file = joinpath(data_save_directory, ekp_filename(cfg, N_ens, rng_idx))
    ekpobj = load(ekp_file)["ekpobj"]
    N_ens = get_N_ens(ekpobj)
    nx = length(truth_params)
    y = get_obs(ekpobj)
    ny = length(y)

    # plot - EKI results
    gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
    p1 = plot(
        range(0, nx - 1, step = 1),
        truth_params,
        label = "solution",
        color = :black,
        linewidth = 4,
        xlabel = "Spatial index",
        ylabel = "Forcing difference (input)",
        left_margin = 15mm,
        bottom_margin = 15mm,
    )

    p2 = plot(
        1:ny,
        y,
        ribbon = sqrt.(diag(get_obs_noise_cov(ekpobj))),
        label = "data",
        color = :black,
        linewidth = 4,
        xlabel = "Spatial index",
        ylabel = "State mean/std output)",
        left_margin = 15mm,
        bottom_margin = 15mm,
    )

    l = @layout [a b]
    plt = plot(p1, p2, layout = l)

    savefig(plt, joinpath(figure_save_directory, "data_$(calib_filename_suffix).png"))
    savefig(plt, joinpath(figure_save_directory, "data_$(calib_filename_suffix).pdf"))

    gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
    p1 = plot(
        range(0, nx - 1, step = 1),
        zeros(nx),
        label = "solution",
        color = :black,
        linewidth = 4,
        xlabel = "Spatial index",
        ylabel = "Forcing difference (input)",
        left_margin = 15mm,
        bottom_margin = 15mm,
    )

    p2 = plot(
        1:ny,
        y,
        ribbon = sqrt.(diag(get_obs_noise_cov(ekpobj))),
        label = "data",
        color = :black,
        linewidth = 4,
        xlabel = "Spatial index",
        ylabel = "State mean/std output)",
        left_margin = 15mm,
        bottom_margin = 15mm,
    )

    p3 = deepcopy(p1)
    p4 = deepcopy(p2)

    plot!(p1, range(0, nx - 1, step = 1), final_params .- truth_params, label = "mean ensemble input", color = :lightgreen, linewidth = 4)
    plot!(p2, 1:length(y), get_g_mean_final(ekpobj), label = "mean ensemble output", color = :lightgreen, linewidth = 4)

    l = @layout [a b]
    plt = plot(p1, p2, layout = l)

    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix).png"))
    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix).pdf"))

    plot!(
        p3,
        range(0, nx - 1, step = 1),
        get_ϕ_final(prior, ekpobj)[:, 1] .- truth_params,
        label = "ensemble inputs",
        color = :lightgreen,
        linewidth = 4,
        linealpha = 0.1,
    )

    plot!(
        p3,
        range(0, nx - 1, step = 1),
        get_ϕ_final(prior, ekpobj)[:, 2:end] .- truth_params,
        label = "",
        color = :lightgreen,
        linewidth = 4,
        linealpha = 0.1,
    )
    plot!(
        p3,
        range(0, nx - 1, step = 1),
        get_ϕ(prior, ekpobj, 1)[:, 1] .- truth_params,
        label = "",
        color = :lightgreen,
        linewidth = 4,
        linealpha = 0.1,
    )

    plot!(
        p3,
        range(0, nx - 1, step = 1),
        get_ϕ(prior, ekpobj, 1)[:, 2:end] .- truth_params,
        label = "",
        color = :lightgreen,
        linewidth = 4,
        linealpha = 0.1,
    )

    plot!(
        p4,
        1:length(y),
        get_g_final(ekpobj)[:, 1],
        color = :lightgreen,
        label = "ensemble outputs",
        linewidth = 4,
        linealpha = 0.1,
    )
    plot!(p4, 1:length(y), get_g_final(ekpobj)[:, 2:end], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)
    plot!(p4, 1:length(y), get_g(ekpobj, 1)[:, 1], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)
    plot!(p4, 1:length(y), get_g(ekpobj, 1)[:, 2:end], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)

    plt = plot(p3, p4, layout = l)

    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix)_full_ens.png"))
    savefig(plt, joinpath(figure_save_directory, "solution_$(calib_filename_suffix)_full_ens.pdf"))

    # plot - UQ results, one figure per k
    for k in k_values
        posterior = posteriors_by_k[k]

        posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...)
        constrained_posterior_samples =
            mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)
        quantiles = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(constrained_posterior_samples)])'

        gr(size = (1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
        p1 = plot(
            range(0, nx - 1, step = 1),
            [truth_params final_params] .- truth_params,
            label = ["solution" "EKI-opt"],
            color = [:black :lightgreen],
            linewidth = 4,
            xlabel = "Spatial index",
            ylabel = "Forcing difference (input)",
            left_margin = 15mm,
            bottom_margin = 15mm,
        )

        plot!(
            p1,
            range(0, nx - 1, step = 1),
            quantiles[:, 2] .- truth_params,
            color = :blue,
            label = "posterior",
            linewidth = 4,
            ribbon = [quantiles[:, 2] - quantiles[:, 1] quantiles[:, 3] - quantiles[:, 2]],
            fillalpha = 0.1,
        )

        figpath = joinpath(figure_save_directory, "posterior_ribbons_$(calib_filename_suffix)_k$(k)")
        savefig(figpath * ".png")
        savefig(figpath * ".pdf")
    end
end
