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

        n_p = length(truth_params_constrained)
        ny  = length(y)

        gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
        p3 = plot(
            range(0, n_p - 1, step = 1),
            zeros(n_p),
            label = "solution",
            color = :black,
            linewidth = 4,
            xlabel = "Parameter index",
            ylabel = "Forcing difference (input)",
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

        plot!(
            p3,
            range(0, n_p - 1, step = 1),
            constrained_push_ensemble[:, 1] .- truth_params_constrained,
            label = "posterior samples",
            color = :lightgreen,
            linewidth = 4,
            linealpha = 0.1,
        )
        plot!(
            p3,
            range(0, n_p - 1, step = 1),
            constrained_push_ensemble[:, 2:end] .- truth_params_constrained,
            label = "",
            color = :lightgreen,
            linewidth = 4,
            linealpha = 0.1,
        )

        plot!(
            p4,
            1:ny,
            G_ens[:, 1],
            label = "pushforward outputs",
            color = :lightgreen,
            linewidth = 4,
            linealpha = 0.1,
        )
        plot!(p4, 1:ny, G_ens[:, 2:end], color = :lightgreen, label = "", linewidth = 4, linealpha = 0.1)

        l = @layout [a b]
        plt = plot(p3, p4, layout = l)

        suffix = case_suffix(cfg, N_ens, rng_idx)
        figure_fn = "pushforward_from_posterior_$(suffix)_k$(k)_full_ens.png"
        savefig(plt, joinpath(data_save_directory, figure_fn))
        savefig(plt, joinpath(data_save_directory, replace(figure_fn, ".png" => ".pdf")))

    end
end        


