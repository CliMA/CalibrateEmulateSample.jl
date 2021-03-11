# Reference the in-tree version of CalibrateEmulateSample on Julias load path
prepend!(LOAD_PATH, [joinpath(@__DIR__, "..", "..")])
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Random
using Distributions
using Statistics
using LinearAlgebra
using CalibrateEmulateSample.GaussianProcessEmulator
using CalibrateEmulateSample.DataStorage

using Plots; pyplot(size=(1500, 700))
Plots.scalefontsizes(1.3)

###############################################################################
#                                                                             #
#   This examples shows how to fit a Gaussian Process regression model        #
#   using GaussianProcess, and how to plot the mean and variance                   #
#                                                                             #
#   Training points: {(x_i, y_i)} (i=1,...,n), where x_i ∈ ℝ ² and y_i ∈ ℝ ²  #
#   The y_i are assumed to be related to the x_i by:                          #
#   y_i = G(x_i) + η, where G is a map from ℝ ² to ℝ ², and η is noise drawn  #
#   from a 2D normal distribution with known mean μ and covariance Σ          #
#                                                                             #
#   Two Gaussian Process models are fit, one that predicts y_i[1] from x_i    #
#   and one that predicts y_i[2] from x_i. Fitting separate models for        #
#   y_i[1] and y_i[2] requires the outputs to be independent - since this     #
#   cannot be assumed to be the case a priori, the training data are          #
#   decorrelated by perfoming a Singular Value Decomposition (SVD) on the     #
#   noise covariance Σ, and each model is then trained in the decorrelated    #
#   space.                                                                    #
#                                                                             #
#   The decorrelation is done automatically when instantiating a `GaussianProcess`,     #
#   but the user can choose to have the `predict` function return its         #
#   predictions in the original space (by setting transform_to_real=true)     # 
#                                                                             #
#   The GaussianProcessEmulator module can be used as a standalone module to fit           #
#   Gaussian Process regression models, but it was originally designed as     #
#   the "Emulate" step of the "Calibrate-Emulate-Sample" framework            #
#   developed at CliMA.                                                       #
#   Further reading on Calibrate-Emulate-Sample:                              #
#       Cleary et al., 2020                                                   #
#       https://arxiv.org/abs/2001.03689                                      #
#                                                                             #
###############################################################################

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where T
    m, n = length(vy), length(vx)
    gx = reshape(repeat(vx, inner = m, outer = 1), m, n)
    gy = reshape(repeat(vy, inner = 1, outer = n), m, n)

    return gx, gy
end

rng_seed = 41
Random.seed!(rng_seed)

output_directory = joinpath(@__DIR__, "output")
if !isdir(output_directory)
    mkdir(output_directory)
end

gppackage = GPJL()
pred_type = YType()

# Generate training data (x-y pairs, where x ∈ ℝ ᵖ, y ∈ ℝ ᵈ)
# x = [x1, x2]: inputs/predictors/features/parameters
# y = [y1, y2]: outputs/predictands/targets/observables
# The observables y are related to the parameters x by: 
# y = G(x1, x2) + η, 
# where G(x1, x2) := [sin(x1) + cos(x2), sin(x1) - cos(x2)], and η ~ N(0, Σ)
n = 50  # number of training points
p = 2   # input dim 
d = 2   # output dim

X = 2.0 * π * rand(p, n)
# G(x1, x2)
g1x = sin.(X[1, :]) .+ cos.(X[2, :])
g2x = sin.(X[1, :]) .- cos.(X[2, :])
gx = zeros(2,n)
gx[1,:] = g1x
gx[2,:] = g2x

# Add noise η
μ = zeros(d) 
Σ = 0.1 * [[0.8, 0.0] [0.0, 0.5]] # d x d
noise_samples = rand(MvNormal(μ, Σ), n) 
# y = G(x) + η
Y = gx .+ noise_samples

#plot training data with and without noise
p1 = plot(X[1,:], X[2,:], g1x, st=:surface, camera=(-30, 30), c=:cividis, 
              xlabel="x1", ylabel="x2",
              zguidefontrotation=90)
figpath = joinpath(output_directory, "GP_test_observed_y1nonoise.png")
savefig(figpath)
linkfig(figpath)

p2 = plot(X[1,:], X[2,:], g2x, st=:surface, camera=(-30, 30), c=:cividis, 
              xlabel="x1", ylabel="x2",
              zguidefontrotation=90)
figpath = joinpath(output_directory, "GP_test_observed_y2nonoise.png")
savefig(figpath)
linkfig(figpath)

p1 = plot(X[1,:], X[2,:], Y[1,:], st=:surface, camera=(-30, 30), c=:cividis, 
              xlabel="x1", ylabel="x2",
              zguidefontrotation=90)
figpath = joinpath(output_directory, "GP_test_observed_y1.png")
savefig(figpath)
linkfig(figpath)

p2 = plot(X[1,:], X[2,:], Y[2,:], st=:surface, camera=(-30, 30), c=:cividis, 
              xlabel="x1", ylabel="x2",
              zguidefontrotation=90)
figpath = joinpath(output_directory, "GP_test_observed_y2.png")
savefig(figpath)
linkfig(figpath)

iopairs = PairedDataContainer(X,Y,data_are_columns=true)
@assert get_inputs(iopairs) == X
@assert get_outputs(iopairs) == Y

# Fit 2D Gaussian Process regression model
# (To be precise: We fit two models, one that predicts y1 from x1 and x2, 
# and one that predicts y2 from x1 and x2)
# Setting GPkernel=nothing leads to the creation of a default squared 
# exponential kernel. 
# Setting noise_learn=true leads to the addition of white noise to the
# kernel
gpobj = GaussianProcess(iopairs, gppackage, GPkernel=nothing, obs_noise_cov=Σ, 
              normalized=true, noise_learn=true, prediction_type=pred_type)

# Plot mean and variance of the predicted observables y1 and y2
# For this, we generate test points on a x1-x2 grid.
n_pts = 100
x1 = range(0.0, stop=2*π, length=n_pts)
x2 = range(0.0, stop=2*π, length=n_pts) 
X1, X2 = meshgrid(x1, x2)
# Input for predict has to be of size N_samples x input_dim
inputs = permutedims(hcat(X1[:], X2[:]),(2,1))

font = Plots.font("Helvetica", 18)
fontdict = Dict(:guidefont=>font, :xtickfont=>font, :ytickfont=>font, 
                :legendfont=>font)
# Predict on the grid points (note that `predict` returns the full
# covariance matrices, not just the variance -- gp_cov is a vector
# of covariance matrices)
gp_mean, gp_cov = GaussianProcessEmulator.predict(gpobj, 
                                                  inputs, 
                                                  transform_to_real=true)
println("end predictions at ", n_pts*n_pts, " points")

println("start plotting...")

#plot predictions
for y_i in 1:d

    gp_var_temp = [diag(gp_cov[j]) for j in 1:length(gp_cov)] # (40000,)
    gp_var = permutedims(vcat([x' for x in gp_var_temp]...),(2,1)) # 2 x 40000
   
    mean_grid = reshape(gp_mean[y_i, :], n_pts, n_pts) # 2 x 40000
    p1 = plot(x1, x2, mean_grid, st=:surface, camera=(-30, 30), c=:cividis, 
              xlabel="x1", ylabel="x2", zlabel="mean of y"*string(y_i),
              zguidefontrotation=90)

    var_grid = reshape(gp_var[y_i, :], n_pts, n_pts)
    p2 = plot(x1, x2, var_grid, st=:surface, camera=(-30, 30), c=:cividis,
              xlabel="x1", ylabel="x2", zlabel="var of y"*string(y_i),
              zguidefontrotation=90)

    plot(p1, p2, layout=(1, 2), legend=false)

    figpath = joinpath(output_directory, "GP_test_y"*string(y_i)*"_predictions.png")
    savefig(figpath)
    linkfig(figpath)
end

# Plot the true components of G(x1, x2)
g1_true = sin.(inputs[1,:]) .+ cos.(inputs[2, :])
g1_true_grid = reshape(g1_true, n_pts, n_pts)
p3 = plot(x1, x2, g1_true_grid, st=:surface, camera=(-30, 30), c=:cividis, 
          xlabel="x1", ylabel="x2", zlabel="sin(x1) + cos(x2)",
          zguidefontrotation=90)
figpath = joinpath(output_directory, "GP_test_true_g1.png")
savefig(figpath)
linkfig(figpath)

g2_true = sin.(inputs[1, :]) .- cos.(inputs[2, :])
g2_true_grid = reshape(g2_true, n_pts, n_pts)
p4 = plot(x1, x2, g2_true_grid, st=:surface, camera=(-30, 30), c=:cividis, 
          xlabel="x1", ylabel="x2", zlabel="sin(x1) - cos(x2)",
          zguidefontrotation=90)
g_true_grids = [g1_true_grid, g2_true_grid]
figpath = joinpath(output_directory, "GP_test_true_g2.png")
savefig(figpath)
linkfig(figpath)


# Plot the difference between the truth and the mean of the predictions
for y_i in 1:d

    # Reshape gp_cov to size N_samples x output_dim
    gp_var_temp = [diag(gp_cov[j]) for j in 1:length(gp_cov)] # (40000,)
    gp_var = permutedims(vcat([x' for x in gp_var_temp]...),(2,1)) # 40000 x 2

    mean_grid = reshape(gp_mean[y_i,:], n_pts, n_pts)
    var_grid = reshape(gp_var[y_i, :], n_pts, n_pts)
    # Compute and plot 1/variance * (truth - prediction)^2
    zlabel = "1/var * (true_y"*string(y_i)*" - predicted_y"*string(y_i)*")^2"
    p5 = plot(x1, x2, 1.0 ./ var_grid .* (g_true_grids[y_i] .- mean_grid).^2, 
              st=:surface, camera=(-30, 30), c=:magma, zlabel=zlabel,
              xlabel="x1", ylabel="x2", 
              zguidefontrotation=90)

    figpath = joinpath(output_directory, "GP_test_y"*string(y_i)*"_difference_truth_prediction.png")
    savefig(figpath)
    linkfig(figpath)
end

Plots.scalefontsizes(1/1.3)
