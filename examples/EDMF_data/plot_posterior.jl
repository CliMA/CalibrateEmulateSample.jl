# Import modules
using StatsPlots
using JLD2
using Dates

# CES 
using CalibrateEmulateSample.ParameterDistributions


# date = Date(year,month,day)

# 2-parameter calibration exp
exp_name = "ent-det-calibration"
date_of_run = Date(2022, 7, 15)

# 5-parameter calibration exp
#exp_name = "ent-det-tked-tkee-stab-calibration"
#date_of_run = Date(2022,7,14)

# Output figure read/write directory
figure_save_directory = joinpath(@__DIR__, "output", exp_name, string(date_of_run))
data_save_directory = joinpath(@__DIR__, "output", exp_name, string(date_of_run))

# load
posterior_filepath = joinpath(data_save_directory, "posterior.jld2")
if !isfile(posterior_filepath)
    LoadError(posterior_filepath * " not found. Please check experiment name and date")
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
density_filepath = joinpath(figure_save_directory, "posterior_dist_comp.png")
transformed_density_filepath = joinpath(figure_save_directory, "posterior_dist_phys.png")
labels = get_name(posterior)

gr(dpi = 300, size = (nparam_plots * 300, nparam_plots * 300))
# lower triangular marginal cornerplot
p = cornerplot(permutedims(posterior_samples, (2, 1)), label = labels, compact = true)
trans_p = cornerplot(permutedims(transformed_posterior_samples, (2, 1)), label = labels, compact = true)

savefig(p, density_filepath)
savefig(trans_p, transformed_density_filepath)
