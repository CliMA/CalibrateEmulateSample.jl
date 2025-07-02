# some context values (from calibrate_spatial_dep.jl)

# some context calues from emulate_sample_spatial_dep.jl
cases = [
    "GP", # SLOW
    "RF-scalar", # diagonalize, train scalar RF, don't asume diag inputs
]
case = cases[2]

# load packages
# CES 

using Random
using JLD2
using Statistics
using LinearAlgebra

using Plots
using Plots.Measures

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

# load
homedir = pwd()
println(homedir)
figure_save_directory = joinpath(homedir, "output/")
data_save_directory = joinpath(homedir, "output/")
data_file = joinpath(data_save_directory, "posterior_$(case).jld2")
truth_params = load(data_file)["truth_params_constrained"]
final_params = load(data_file)["final_params_constrained"]
posterior = load(data_file)["posterior"]


ekp_file = joinpath(data_save_directory, "ekp_spatial_dep.jld2")
ekpobj = load(ekp_file)["ekpobj"]
N_ens = get_N_ens(ekpobj)
nx = length(truth_params)
y = get_obs(ekpobj)
ny = length(y)
# get samples and quantiles from posterior
param_names = get_name(posterior)
posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
constrained_posterior_samples =
    mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

quantiles = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(constrained_posterior_samples)])' # rows are quantiles for row of posterior samples


# plot - EKI results
gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
p1 = plot(
    range(0, nx - 1, step = 1),
    [truth_params final_params],
    label = ["solution" "EKI"],
    color = [:black :lightgreen],
    linewidth = 4,
    xlabel = "Spatial index",
    ylabel = "Forcing (input)",
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
    xticks = (Int.(0:10:ny), [0, 10, 20, 30, (40, 0), 10, 20, 30, 40]),
)
plot!(p2, 1:length(y), get_g_mean_final(ekpobj), label = "mean-final-output", color = :lightgreen, linewidth = 4)

l = @layout [a b]
plt = plot(p1, p2, layout = l)

savefig(plt, figure_save_directory * "solution_spatial_dep_ens$(N_ens).png")
savefig(plt, figure_save_directory * "solution_spatial_dep_ens$(N_ens).pdf")


# plot - UQ results

gr(size = (1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
p1 = plot(
    range(0, nx - 1, step = 1),
    [truth_params final_params],
    label = ["solution" "EKI-opt"],
    color = [:black :lightgreen],
    linewidth = 4,
    xlabel = "Spatial index",
    ylabel = "Forcing (input)",
    left_margin = 15mm,
    bottom_margin = 15mm,
)

plot!(
    p1,
    range(0, nx - 1, step = 1),
    quantiles[:, 2], # median of all vals
    color = :blue,
    label = "posterior",
    linewidth = 4,
    ribbon = [quantiles[:, 2] - quantiles[:, 1] quantiles[:, 3] - quantiles[:, 2]],
    fillalpha = 0.1,
)


figpath = joinpath(figure_save_directory, "posterior_ribbons_" * case)
savefig(figpath * ".png")
savefig(figpath * ".pdf")
