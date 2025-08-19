# some context values (from calibrate_spatial_dep.jl)

# some context calues from emulate_sample_spatial_dep.jl
cases = ["GP", "RF-scalar"]
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

cases = ["GP", "RF-scalar"]
case = cases[2]

# load
homedir = pwd()
println(homedir)
figure_save_directory = joinpath(homedir, "output/")
data_save_directory = joinpath(homedir, "output/")
data_file = joinpath(data_save_directory, "posterior_$(case).jld2")
truth_params = load(data_file)["truth_params_constrained"]
final_params = load(data_file)["final_params_constrained"]
posterior = load(data_file)["posterior"]

prior_file = joinpath(data_save_directory, "priors_spatial_dep.jld2")
prior = load(prior_file)["prior"]

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
    truth_params,
    label = "solution",
    color = :black,
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

l = @layout [a b]
plt = plot(p1, p2, layout = l)

savefig(plt, figure_save_directory * "data_spatial_dep.png")
savefig(plt, figure_save_directory * "data_spatial_dep.pdf")

p3 = deepcopy(p1)
p4 = deepcopy(p2)

plot!(p1, range(0, nx - 1, step = 1), final_params, label = "mean ensemble input", color = :lightgreen, linewidth = 4)
plot!(p2, 1:length(y), get_g_mean_final(ekpobj), label = "mean ensemble output", color = :lightgreen, linewidth = 4)

l = @layout [a b]
plt = plot(p1, p2, layout = l)

savefig(plt, figure_save_directory * "solution_spatial_dep_ens$(N_ens).png")
savefig(plt, figure_save_directory * "solution_spatial_dep_ens$(N_ens).pdf")

#
plot!(
    p3,
    range(0, nx - 1, step = 1),
    get_ϕ_final(prior, ekpobj)[:, 1],
    label = "ensemble inputs",
    color = :lightgreen,
    linewidth = 4,
    linealpha = 0.1,
)

plot!(
    p3,
    range(0, nx - 1, step = 1),
    get_ϕ_final(prior, ekpobj)[:, 2:end],
    label = "",
    color = :lightgreen,
    linewidth = 4,
    linealpha = 0.1,
)
plot!(
    p3,
    range(0, nx - 1, step = 1),
    get_ϕ(prior, ekpobj, 1)[:, 1],
    label = "",
    color = :lightgreen,
    linewidth = 4,
    linealpha = 0.1,
)

plot!(
    p3,
    range(0, nx - 1, step = 1),
    get_ϕ(prior, ekpobj, 1)[:, 2:end],
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

savefig(plt, figure_save_directory * "solution_spatial_dep_full_ens$(N_ens).png")
savefig(plt, figure_save_directory * "solution_spatial_dep_full_ens$(N_ens).pdf")

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

##########################
cases = ["GP", "RF-scalar"]

# load
homedir = pwd()
println(homedir)
data_file_GP = joinpath(data_save_directory, "posterior_$(cases[1]).jld2")
data_file_RF = joinpath(data_save_directory, "posterior_$(cases[2]).jld2")
truth_params_GP = load(data_file_GP)["truth_params_constrained"]
final_params_GP = load(data_file_GP)["final_params_constrained"]
posterior_GP = load(data_file_GP)["posterior"]
truth_params_RF = load(data_file_RF)["truth_params_constrained"]
final_params_RF = load(data_file_RF)["final_params_constrained"]
posterior_RF = load(data_file_RF)["posterior"]

ekp_file = joinpath(data_save_directory, "ekp_spatial_dep.jld2")
ekpobj = load(ekp_file)["ekpobj"]
N_ens = get_N_ens(ekpobj)
nx = length(truth_params)
y = get_obs(ekpobj)
ny = length(y)
# get samples and quantiles from posterior
param_names_GP = get_name(posterior_GP)
posterior_samples_GP = vcat([get_distribution(posterior_GP)[name] for name in get_name(posterior_GP)]...) #samples are columns
constrained_posterior_samples_GP =
    mapslices(x -> transform_unconstrained_to_constrained(posterior_GP, x), posterior_samples_GP, dims = 1)

quantiles_GP = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(constrained_posterior_samples_GP)])' # rows are quantiles for row of posterior samples

param_names_RF = get_name(posterior_RF)
posterior_samples_RF = vcat([get_distribution(posterior_RF)[name] for name in get_name(posterior_RF)]...) #samples are columns
constrained_posterior_samples_RF =
    mapslices(x -> transform_unconstrained_to_constrained(posterior_RF, x), posterior_samples_RF, dims = 1)

quantiles_RF = reduce(hcat, [quantile(row, [0.05, 0.5, 0.95]) for row in eachrow(constrained_posterior_samples_RF)])' # rows are quantiles for row of posterior samples

# plot - UQ results - both

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
    quantiles_GP[:, 2], # median of all vals
    color = :blue,
    label = "GP posterior",
    linewidth = 4,
    ribbon = [quantiles_GP[:, 2] - quantiles_GP[:, 1] quantiles_GP[:, 3] - quantiles_GP[:, 2]],
    linealpha = 0.5,
    fillalpha = 0.1,
)

plot!(
    p1,
    range(0, nx - 1, step = 1),
    quantiles_RF[:, 2], # median of all vals
    color = :red,
    label = "posterior_RF",
    linewidth = 4,
    ribbon = [quantiles_RF[:, 2] - quantiles_RF[:, 1] quantiles_RF[:, 3] - quantiles_RF[:, 2]],
    linealpha = 0.5,
    fillalpha = 0.1,
)


figpath = joinpath(figure_save_directory, "posterior_ribbons_both")
savefig(figpath * ".png")
savefig(figpath * ".pdf")
