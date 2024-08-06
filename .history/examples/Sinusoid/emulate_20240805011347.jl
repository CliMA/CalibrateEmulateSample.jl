# # [Emulate](@id sinusoid-example)

# In this example we have a model that produces a sinusoid
# ``f(A, v) = A \sin(\phi + t) + v, \forall t \in [0,2\pi]``, with a random
# phase ``\phi``. We want to quantify uncertainties on parameters ``A`` and ``v``,
# given noisy observations of the model output.
# Previously, in the calibration step, we started with an initial guess of these
# parameters and used Ensemble Kalman Inversion to iteratively update our estimates.
# Quantifying uncertainties around these estimates requires many model evaluations
# and can quickly become expensive.
# On this page, we will see how we can build an emulator to help us with
# this task.

# First, we load the packages we need:
using LinearAlgebra, Random

using Distributions, Plots, Plots.PlotMeasures
using JLD2

using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.EnsembleKalmanProcesses

const CES = CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses

include("sinusoid_setup.jl")           # This file defines the model G(θ)


# Load the ensembles generated in the calibration step.
example_directory = @__DIR__
data_save_directory = joinpath(example_directory, "output")
# Get observations, true parameters and observation noise
obs_file = joinpath(data_save_directory, "observations.jld2")
y_obs = load(obs_file)["y_obs"]
theta_true = load(obs_file)["theta_true"]
Γ = load(obs_file)["Γ"]
# Get EKI and prior
calibrate_file = joinpath(data_save_directory, "calibrate_results.jld2")
ensemble_kalman_process = load(calibrate_file)["eki"]
prior = load(calibrate_file)["prior"]
N_ensemble = load(calibrate_file)["N_ensemble"]
N_iterations = load(calibrate_file)["N_iterations"]
# Get random number generator to start where we left off
rng = load(calibrate_file)["rng"]


# We ran Ensemble Kalman Inversion with an ensemble size of 10 for 5
# iterations. This generated a total of 50 input output pairs from our model.
# We will use these samples to train an emulator. The EKI samples make a suitable
# dataset for training an emulator because in the first iteration, the ensemble parameters
# are spread out according to the prior, meaning they cover the full support of the
# parameter space. This is important for building an emulator that can be evaluated anywhere
# in this space. In later iterations, the ensemble parameters are focused around the truth.
# This means the emulator that will be more accurate around this region.


## Build Emulators
# We will build two types of emulator here for comparison: Gaussian processes and Random
# Features. First, set up the data in the correct format. CalibrateEmulateSample.jl uses
# a paired data container that matches the inputs (in the unconstrained space) to the outputs:
input_output_pairs = CES.Utilities.get_training_points(ensemble_kalman_process, N_iterations)
unconstrained_inputs = CES.Utilities.get_inputs(input_output_pairs)
inputs = Emulators.transform_unconstrained_to_constrained(prior, unconstrained_inputs)
outputs = CES.Utilities.get_outputs(input_output_pairs)

# Gaussian process
# We will set up a basic Gaussian process emulator using either ScikitLearn.jl or GaussianProcesses.jl.
# See the Gaussian process page for more information and details about kernel choices.
gppackage = Emulators.AGPJL()
gauss_proc = Emulators.GaussianProcess(gppackage, noise_learn = false)

# Build emulator with data
emulator_gp = Emulator(gauss_proc, input_output_pairs, normalize_inputs = true, obs_noise_cov = Γ)
optimize_hyperparameters!(emulator_gp)

# We have built the Gaussian process emulator and we can now use it for prediction. We will validate the emulator
# performance soon, but first, build a random features emulator for comparison.

## Random Features

# An alternative to Gaussian process for the emulator
# We have two input dimensions and two output dimensions.
input_dim = 2
output_dim = 2

# Select number of features
n_features = 60
nugget = 1e-9

# Create random features
# Here we use a vector random features set up because we find it performs similarly well to
# the GP emulator, but there are many options that could be tested
kernel_structure = NonseparableKernel(LowRankFactor(2, nugget))
optimizer_options = Dict(
    "n_ensemble" => 50,
    "cov_sample_multiplier" => 10,
    "scheduler" => EKP.DataMisfitController(on_terminate = "continue"),
    "n_iteration" => 50,
    "rng" => rng,
    "verbose" => true,
)
random_features = VectorRandomFeatureInterface(
    n_features,
    input_dim,
    output_dim,
    rng = rng,
    kernel_structure = kernel_structure,
    optimizer_options = optimizer_options,
)
emulator_random_features =
    Emulator(random_features, input_output_pairs, normalize_inputs = true, obs_noise_cov = Γ, decorrelate = false)
optimize_hyperparameters!(emulator_random_features)


## Emulator Validation
# Now we will validate both GP and RF emulators and compare them against the ground truth, G(θ).
# Note this is only possible in our example because our true model, G(θ), is cheap to evaluate.
# In more complex systems, we would have limited data to validate emulator performance with.
# Set up mesh to cover parameter space
N_grid = 50
amp_range = range(0.6, 4, length = N_grid)
vshift_range = range(-6, 10, length = N_grid)

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where {T}
    m, n = length(vy), length(vx)
    gx = reshape(repeat(vx, inner = m, outer = 1), m, n)
    gy = reshape(repeat(vy, inner = 1, outer = n), m, n)

    return gx, gy
end

X1, X2 = meshgrid(amp_range, vshift_range)
# Input for predict has to be of size  input_dim x N_samples
input_grid = permutedims(hcat(X1[:], X2[:]), (2, 1))
# First, predict with true model
g_true = hcat([G(input_grid[:, i]; rng = rng) for i in 1:size(input_grid)[2]]...)
g_true_grid = reshape(g_true, (output_dim, N_grid, N_grid))

# Next, predict with emulators. We first need to transform to the unconstrained space.
input_grid_unconstrained = Emulators.transform_constrained_to_unconstrained(prior, input_grid)
gp_mean, gp_cov = Emulators.predict(emulator_gp, input_grid_unconstrained, transform_to_real = true)
rf_mean, rf_cov = Emulators.predict(emulator_random_features, input_grid_unconstrained, transform_to_real = true)

# Reshape into (output_dim x 50 x 50) grid
output_dim = 2
gp_grid = reshape(gp_mean, (output_dim, N_grid, N_grid))
rf_grid = reshape(rf_mean, (output_dim, N_grid, N_grid))

# Convert cov matrix into std and reshape
gp_std = [sqrt.(diag(gp_cov[j])) for j in 1:length(gp_cov)]
gp_std_grid = reshape(permutedims(reduce(vcat, [x' for x in gp_std]), (2, 1)), (output_dim, N_grid, N_grid))
rf_std = [sqrt.(diag(rf_cov[j])) for j in 1:length(rf_cov)]
rf_std_grid = reshape(permutedims(reduce(vcat, [x' for x in rf_std]), (2, 1)), (output_dim, N_grid, N_grid))

## Plot
# First, we will plot the ground truth. We have 2 parameters and 2 outputs, so we will create a contour plot
# for each output to show how they vary against the two inputs.
range_clims = (0, 8)
mean_clims = (-6, 10)

p1 = contour(
    amp_range,
    vshift_range,
    g_true_grid[1, :, :];
    fill = true,
    clims = range_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "True Sinusoid Range",
)
p2 = contour(
    amp_range,
    vshift_range,
    g_true_grid[2, :, :];
    fill = true,
    clims = mean_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "True Sinusoid Mean",
)
p = plot(p1, p2, size = (600, 300), layout = (1, 2), guidefontsize = 12, tickfontsize = 10, legendfontsize = 10)
savefig(p, joinpath(data_save_directory, "sinusoid_groundtruth_contours.png"))
# The first panel shows how the range varies with respect to the two parameters in the Gaussian process
# emulator. The contours show the range is mostly dependent on the amplitude, with little variation with
# respect to the vertical shift. The second panel shows how the mean varies with the respect to the two
# parameters and is mostly dependent on the vertical shift. This result makes sense for our model setup.
# Next, we recreate the same plot with the emulators. Ideally, the results should look similar.

# Plot GP emulator contours
p1 = contour(
    amp_range,
    vshift_range,
    gp_grid[1, :, :];
    fill = true,
    clims = range_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "GP Sinusoid Range",
)
# We will also overlay the training data points from the EKI. The colors of the points should agree with
# the contours.
plot!(inputs[1, :], inputs[2, :]; seriestype = :scatter, zcolor = outputs[1, :], label = :false)
p2 = contour(
    amp_range,
    vshift_range,
    gp_grid[2, :, :];
    fill = true,
    clims = mean_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "GP Sinusoid Mean",
)
plot!(inputs[1, :], inputs[2, :]; seriestype = :scatter, zcolor = outputs[2, :], label = :false)
p = plot(
    p1,
    p2,
    right_margin = 3mm,
    bottom_margin = 3mm,
    size = (600, 300),
    layout = (1, 2),
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
)
savefig(p, joinpath(data_save_directory, "sinusoid_GP_emulator_contours.png"))

# Plot RF emulator contours
p1 = contour(
    amp_range,
    vshift_range,
    rf_grid[1, :, :];
    fill = true,
    clims = range_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "RF Sinusoid Range",
)
plot!(inputs[1, :], inputs[2, :]; seriestype = :scatter, zcolor = outputs[1, :], label = :false)
p2 = contour(
    amp_range,
    vshift_range,
    rf_grid[2, :, :];
    fill = true,
    clims = mean_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "RF Sinusoid Mean",
)
plot!(inputs[1, :], inputs[2, :]; seriestype = :scatter, zcolor = outputs[2, :], label = :false)
p = plot(
    p1,
    p2,
    right_margin = 3mm,
    bottom_margin = 3mm,
    size = (600, 300),
    layout = (1, 2),
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
)
savefig(p, joinpath(data_save_directory, "sinusoid_RF_emulator_contours.png"))

# Both the GP and RF emulator give similar results to the ground truth G(θ), indicating they are correctly
# learning the relationships between the parameters and the outputs. We also see the contours agree with the
# colors of the training data points.

# Next, we plot uncertainty estimates from the GP and RF emulators
# Plot GP std estimates
range_std_clims = (0, 2)
mean_std_clims = (0, 1)

p1 = contour(
    amp_range,
    vshift_range,
    gp_std_grid[1, :, :];
    c = :cividis,
    fill = true,
    clims = range_std_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "GP 1σ in Sinusoid Range",
)
p2 = contour(
    amp_range,
    vshift_range,
    gp_std_grid[2, :, :];
    c = :cividis,
    fill = true,
    clims = mean_std_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "GP 1σ in Sinusoid Mean",
)
p = plot(
    p1,
    p2,
    size = (600, 300),
    layout = (1, 2),
    right_margin = 3mm,
    bottom_margin = 3mm,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
)
savefig(p, joinpath(data_save_directory, "sinusoid_GP_emulator_std_contours.png"))


# Plot RF std estimates
p1 = contour(
    amp_range,
    vshift_range,
    rf_std_grid[1, :, :];
    c = :cividis,
    fill = true,
    clims = range_std_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "RF 1σ in Sinusoid Range",
)
p2 = contour(
    amp_range,
    vshift_range,
    rf_std_grid[2, :, :];
    c = :cividis,
    fill = true,
    clims = mean_std_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "RF 1σ in Sinusoid Mean",
)
p = plot(p1, p2, size = (600, 300), layout = (1, 2), guidefontsize = 12, tickfontsize = 10, legendfontsize = 10)
savefig(p, joinpath(data_save_directory, "sinusoid_RF_emulator_std_contours.png"))
# The GP and RF uncertainty predictions are similar and show lower uncertainties around the region of interest
# where we have more training points.

# Finally, we should validate how accurate the emulators are by looking at the absolute difference between emulator
# predictions and the ground truth.
gp_diff_grid = abs.(gp_grid - g_true_grid)
range_diff_clims = (0, 1)
mean_diff_clims = (0, 1)
p1 = contour(
    amp_range,
    vshift_range,
    gp_diff_grid[1, :, :];
    c = :cividis,
    fill = true,
    clims = range_diff_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "GP error in Sinusoid Range",
)
p2 = contour(
    amp_range,
    vshift_range,
    gp_diff_grid[2, :, :];
    c = :cividis,
    fill = true,
    clims = mean_diff_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "GP error in Sinusoid Mean",
)
p = plot(p1, p2, size = (600, 300), layout = (1, 2), guidefontsize = 12, tickfontsize = 10, legendfontsize = 10)
savefig(p, joinpath(data_save_directory, "sinusoid_GP_errors_contours.png"))

rf_diff_grid = abs.(rf_grid - g_true_grid)
p1 = contour(
    amp_range,
    vshift_range,
    rf_diff_grid[1, :, :];
    c = :cividis,
    fill = true,
    clims = range_diff_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "RF error in Sinusoid Range",
)
p2 = contour(
    amp_range,
    vshift_range,
    rf_diff_grid[2, :, :];
    c = :cividis,
    fill = true,
    clims = mean_diff_clims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
    title = "RF error in Sinusoid Mean",
)
p = plot(p1, p2, size = (600, 300), layout = (1, 2), guidefontsize = 12, tickfontsize = 10, legendfontsize = 10)
savefig(p, joinpath(data_save_directory, "sinusoid_RF_errors_contours.png"))

# Here, we want the emulator to show the low errors in the region around the true parameter values near θ = (3, 6),
# as this is the region that we will be sampling in the next step. This appears to the be case for both
# outputs and both emulators.


println(mean(gp_diff_grid, dims = (2, 3)))
println(mean(rf_diff_grid, dims = (2, 3)))


save(
    joinpath(data_save_directory, "emulators.jld2"),
    "emulator_gp",
    emulator_gp,
    "emulator_random_features",
    emulator_random_features,
    "prior",
    prior,
    "rng",
    rng,
)
