# Import modules
ENV["GKSwstype"] = "100"
using Plots

using CairoMakie, PairPlots
using JLD2
using Dates

# CES 
using CalibrateEmulateSample.ParameterDistributions

#####
# Creates 1 plots: One for a specific case, One with 2 cases, and One with all cases (final case being the prior).


# date = Date(year,month,day)

# 2-parameter calibration exp
#exp_name = "ent-det-calibration"
#date_of_run = Date(2023, 10, 5)

# 5-parameter calibration exp
exp_name = "ent-det-tked-tkee-stab-calibration"
date_of_run = Date(2024, 06, 14)

# Output figure read/write directory
figure_save_directory = joinpath(@__DIR__, "output", exp_name, string(date_of_run))
data_save_directory = joinpath(@__DIR__, "output", exp_name, string(date_of_run))

#case:
cases = [
    "GP", # diagonalize, train scalar GP, assume diag inputs
    "RF-prior",
    "RF-vector-svd-nonsep",
]
case_rf = cases[3]

# load
posterior_filepath = joinpath(data_save_directory, "$(case_rf)_posterior.jld2")
if !isfile(posterior_filepath)
    throw(ArgumentError(posterior_filepath * " not found. Please check experiment name and date"))
else
    println("Loading posterior distribution from: " * posterior_filepath)
    posterior = load(posterior_filepath)["posterior"]
end
# get samples explicitly (may be easier to work with)
posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
transformed_posterior_samples =
    mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

# histograms
nparam_plots = sum(get_dimensions(posterior)) - 1
density_filepath = joinpath(figure_save_directory, "$(case_rf)_posterior_dist_comp.png")
transformed_density_filepath = joinpath(figure_save_directory, "$(case_rf)_posterior_dist_phys.png")
labels = get_name(posterior)

burnin = 50_000

data_rf = (; [(Symbol(labels[i]), posterior_samples[i, burnin:end]) for i in 1:length(labels)]...)
transformed_data_rf =
    (; [(Symbol(labels[i]), transformed_posterior_samples[i, burnin:end]) for i in 1:length(labels)]...)

p = pairplot(data_rf => (PairPlots.Contourf(sigmas = 1:1:3),))
trans_p = pairplot(transformed_data_rf => (PairPlots.Contourf(sigmas = 1:1:3),))

save(density_filepath, p)
save(transformed_density_filepath, trans_p)

#
#
#

case_gp = cases[1]
# load
posterior_filepath = joinpath(data_save_directory, "$(case_gp)_posterior.jld2")
if !isfile(posterior_filepath)
    throw(ArgumentError(posterior_filepath * " not found. Please check experiment name and date"))
else
    println("Loading posterior distribution from: " * posterior_filepath)
    posterior = load(posterior_filepath)["posterior"]
end
# get samples explicitly (may be easier to work with)
posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
transformed_posterior_samples =
    mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

# histograms
nparam_plots = sum(get_dimensions(posterior)) - 1
density_filepath = joinpath(figure_save_directory, "$(case_rf)_$(case_gp)_posterior_dist_comp.png")
transformed_density_filepath = joinpath(figure_save_directory, "$(case_rf)_$(case_gp)_posterior_dist_phys.png")
labels = get_name(posterior)
data_gp = (; [(Symbol(labels[i]), posterior_samples[i, burnin:end]) for i in 1:length(labels)]...)
transformed_data_gp =
    (; [(Symbol(labels[i]), transformed_posterior_samples[i, burnin:end]) for i in 1:length(labels)]...)
#
#
#
gp_smoothing = 1 # >1 = smoothing KDE in plotting

p = pairplot(
    data_rf => (PairPlots.Contourf(sigmas = 1:1:3),),
    data_gp => (PairPlots.Contourf(sigmas = 1:1:3, bandwidth = gp_smoothing),),
)
trans_p = pairplot(
    transformed_data_rf => (PairPlots.Contourf(sigmas = 1:1:3),),
    transformed_data_gp => (PairPlots.Contourf(sigmas = 1:1:3, bandwidth = gp_smoothing),),
)

save(density_filepath, p)
save(transformed_density_filepath, trans_p)



# Finally include the prior too
case_prior = cases[2]
# load
posterior_filepath = joinpath(data_save_directory, "$(case_prior)_posterior.jld2")
if !isfile(posterior_filepath)
    throw(ArgumentError(posterior_filepath * " not found. Please check experiment name and date"))
else
    println("Loading posterior distribution from: " * posterior_filepath)
    posterior = load(posterior_filepath)["posterior"]
end
# get samples explicitly (may be easier to work with)
posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
transformed_posterior_samples =
    mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

# histograms
nparam_plots = sum(get_dimensions(posterior)) - 1
density_filepath = joinpath(figure_save_directory, "all_posterior_dist_comp.png")
transformed_density_filepath = joinpath(figure_save_directory, "all_posterior_dist_phys.png")
labels = get_name(posterior)

data_prior = (; [(Symbol(labels[i]), posterior_samples[i, burnin:end]) for i in 1:length(labels)]...)
transformed_data_prior =
    (; [(Symbol(labels[i]), transformed_posterior_samples[i, burnin:end]) for i in 1:length(labels)]...)
#
#
#

p = pairplot(
    data_rf => (PairPlots.Contourf(sigmas = 1:1:3),),
    data_gp => (PairPlots.Contourf(sigmas = 1:1:3, bandwidth = gp_smoothing),),
    data_prior => (PairPlots.Scatter(),),
)
trans_p = pairplot(
    transformed_data_rf => (PairPlots.Contourf(sigmas = 1:1:3),),
    transformed_data_gp => (PairPlots.Contourf(sigmas = 1:1:3, bandwidth = gp_smoothing),),
    transformed_data_prior => (PairPlots.Scatter(),),
)

save(density_filepath, p)
save(transformed_density_filepath, trans_p)

density_filepath = joinpath(figure_save_directory, "posterior_dist_comp.pdf")
transformed_density_filepath = joinpath(figure_save_directory, "posterior_dist_phys.pdf")
save(density_filepath, p)
save(transformed_density_filepath, trans_p)
