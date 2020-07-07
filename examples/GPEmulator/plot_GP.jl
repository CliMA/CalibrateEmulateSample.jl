# Import modules
using Random
using GaussianProcesses
using Distributions
using Statistics
using Plots; pyplot(size = (1000, 400))
using LinearAlgebra
using CalibrateEmulateSample.GPEmulator

###############################################################################
#                                                                             #
#   This examples shows how to fit a Gaussian Process regression model to     #
#   2D input data, and how to plot the mean and variance                      #
#                                                                             #
###############################################################################

function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where T
    m, n = length(vy), length(vx)
    gx = reshape(repeat(vx, inner = m, outer = 1), m, n)
    gy = reshape(repeat(vy, inner = 1, outer = n), m, n)

    return gx, gy
end

gppackage = GPJL()
pred_type = YType()

# Seed for pseudo-random number generator
rng_seed = 41
Random.seed!(rng_seed)


# Generate training data (x-y pairs, where x ∈ ℝ ᵖ, y ∈ ℝ ᵈ)
# x = [x1, x2]: inputs/predictors/features/parameters
# y = [y1, y2]: outputs/predictands/targets/observables
# The observables y are related to the parameters x by: 
# y = G(x1, x2) + η, 
# where G(x1, x2) := [sin(x1) + cos(x2), sin(x1) - cos(x2)], and η ~ N(0, Σ)
n = 100 # number of training points
p = 2   # input dim 
d = 2   # output dim
#X = 2.0 * π * rand(n, p)
x1 = range(0.0, stop=2*π, length=convert(Int64,sqrt(n)))
x2 = range(0.0, stop=2*π, length=convert(Int64,sqrt(n)))
X1, X2 = meshgrid(x1, x2)

# Input for predict needs to be N_samples x N_parameters
X = hcat(X1[:], X2[:]) 
μ = zeros(d) 
Σ = 0.1 * [[0.8, 0.0] [0.0, 0.5]] # d x d

g1x = sin.(X[:, 1]) .+ cos.(X[:, 2])
g2x = sin.(X[:, 1]) .- cos.(X[:, 2])
noise_samples = rand(MvNormal(μ, Σ), n)' 
gx = [g1x g2x]
Y = gx .+ noise_samples

truth_cov = Σ
sample_truth_cov = cov(noise_samples, dims=1) 


# Fit 2D Gaussian Process regression model
# (To be precise, we fit two models, one that predicts y1 from x1 and x2, 
# and one that predicts y2 from x1 and x2)
gpobj = GPObj(X, Y, truth_cov, gppackage, GPkernel=nothing, 
              normalized=false, noise_learn=false, prediction_type=pred_type)

# Plot mean and variance of the predicted observables y1 and y2
# For this, we generate test points on a x1-x2 grid
n_pts = 50
x1 = range(0.0, stop=2*π, length=n_pts)
x2 = range(0.0, stop=2*π, length=n_pts) 
X1, X2 = meshgrid(x1, x2)

# Input for predict needs to be N_samples x N_parameters
inputs = hcat(X1[:], X2[:]) 

# Get predictions on the grid
gp_mean, gp_var = GPEmulator.predict(gpobj, inputs, transform_to_real=true)

for y_i in 1:d
    mean_grid = reshape(gp_mean[y_i, :], n_pts, n_pts)
    p1 = plot(x1, x2, mean_grid, st=:surface, camera=(-30, 30), c=:cividis, 
              xlabel="x1", ylabel="x2", zlabel="mean of y"*string(y_i))

    var_grid = reshape(gp_var[y_i, :], n_pts, n_pts)
    p2 = plot(x1, x2, var_grid, st=:surface, camera=(-30, 30), c=:cividis,
              xlabel="x1", ylabel="x2", zlabel="var of y"*string(y_i))

    plot(p1, p2, layout=(1, 2), legend=false)
    savefig("GP_test_y"*string(y_i)*".png")
end

# Make plots of the true components of G(x1, x2)
g1_true = sin.(inputs[:, 1]) .+ cos.(inputs[:, 2])
g1_grid = reshape(g1_true, n_pts, n_pts)
p3 = plot(x1, x2, g1_grid, st=:surface, camera=(-30, 30), c=:cividis, 
          xlabel="x1", ylabel="x2", zlabel="sin(x1) + 0.5*x2")
savefig("GP_test_true_g1.png")

g2_true = sin.(inputs[:, 1]) .- cos.(inputs[:, 2])
g2_grid = reshape(g2_true, n_pts, n_pts)
p4 = plot(x1, x2, g2_grid, st=:surface, camera=(-30, 30), c=:cividis, 
          xlabel="x1", ylabel="x2", zlabel="sin(x1) + 0.5*x2")
savefig("GP_test_true_g2.png")
