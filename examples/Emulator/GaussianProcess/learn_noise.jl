# Reference the in-tree version of CalibrateEmulateSample on Julias load path

# Import modules
using Random
using Distributions
using Statistics
using LinearAlgebra
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers

###############################################################################
#                                                                             #
#   This examples shows how to fit a Gaussian Process (GP) regression model   #
#   using Emulator/GaussianProcess and demonstrates how the GaussianProcess's built-in Singular       #
#   Value Decomposition (SVD) decorrelates the training data and maps into    #
#   a space where the covariance of the observational noise is the identity.  #
#                                                                             #
#   When training a GP, the hyperparameters of its kernel (or covariance      #
#   function) are optimized with respect to the given input-output pairs.     #
#   When training a separate GP for each output component (as is usually      #
#   done for multidimensional output, rather than fitting one multi-output    #
#   model), one assumes that their covariance functions are independent.      #
#   The SVD transformation ensures that this is a valid assumption.           #
#                                                                             #
#   The decorrelation by the SVD is done automatically when instantiating     #
#   a `GaussianProcess`, but the user can choose to have the `predict` function         #
#   return its predictions in the original space (by setting                  #
#   transform_to_real=true)                                                   # 
#                                                                             #
#   Training points: {(x_i, y_i)} (i=1,...,n), where x_i ∈ ℝ ² and y_i ∈ ℝ ²  #
#   The y_i are assumed to be related to the x_i by:                          #
#   y_i = G(x_i) + η, where G is a map from ℝ ² to ℝ ², and η is noise drawn  #
#   from a 2D normal distribution with known mean μ and covariance Σ          #
#   Two Gaussian Process models are fit, one that predicts y_i[1] from x_i    #
#   and one that predicts y_i[2] from x_i.                                    #
#                                                                             #
#   The GaussianProcess module can be used as a standalone module to fit      #
#   Gaussian Process regression models, but it was originally designed as     #
#   the "Emulate" step of the "Calibrate-Emulate-Sample" framework            #
#   developed at CliMA.                                                       #
#   Further reading on Calibrate-Emulate-Sample:                              #
#       Cleary et al., 2020                                                   #
#       https://arxiv.org/abs/2001.03689                                      #
#                                                                             #
###############################################################################

rng_seed = 41
Random.seed!(rng_seed)

gppackage = GPJL()
prediction_type = YType()
gauss_proc_1 = GaussianProcess(
    gppackage;
    kernel = nothing, # use default squared exponential kernel
    prediction_type = prediction_type,
    noise_learn = true,
)

# Generate training data (x-y pairs, where x ∈ ℝ ᵖ, y ∈ ℝ ᵈ)
# x = [x1, x2]: inputs/predictors/features/parameters
# y = [y1, y2]: outputs/predictands/targets/observables
# The observables y are related to the parameters x by: 
# y = G(x1, x2) + η, 
# where G(x1, x2) := [sin(x1) + cos(x2), sin(x1) - cos(x2)], and η ~ N(0, Σ)
n = 200  # number of training points
p = 2   # input dim 
d = 2   # output dim

X = 2.0 * π * rand(p, n)

# G(x1, x2)
g1x = sin.(X[1, :]) .+ cos.(X[2, :])
g2x = sin.(X[1, :]) .- cos.(X[2, :])
gx = zeros(2, n)
gx[1, :] = g1x
gx[2, :] = g2x
# Add noise η
μ = zeros(d)
Σ = 0.1 * [[0.8, 0.2] [0.2, 0.5]] # d x d
noise_samples = rand(MvNormal(μ, Σ), n)

# y = G(x) + η
Y = gx .+ noise_samples

# Fit 2D Gaussian Process regression model
# (To be precise: We fit two models, one that predicts y1 from x1 and x2, and 
# one that predicts y2 from x1 and x2)
# Setting GPkernel=nothing leads to the creation of a default squared 
# exponential kernel. 
# Setting noise_learn=true leads to the addition of white noise to the kernel, 
# whose hyperparameter -- the signal variance, or "noise" -- will be learned 
# in the training phase via an optimization procedure.
# Because of the SVD transformation applied to the output, we expect the 
# learned noise to be close to 1.
iopairs = PairedDataContainer(X, Y, data_are_columns = true)
@assert get_inputs(iopairs) == X
@assert get_outputs(iopairs) == Y

emulator_1 = Emulator(gauss_proc_1, iopairs, obs_noise_cov = Σ, normalize_inputs = true)
println("build GP with ", n, " training points")
#optimize the hyperparameters to best fit the GP.
optimize_hyperparameters!(emulator_1)
println("GP trained")

# gpobj1 = GaussianProcess(iopairs, gppackage, GPkernel=nothing, obs_noise_cov=Σ,
#                normalized=true, noise_learn=true, prediction_type=pred_type)

println("\n-----------")
println("Results of training Gaussian Process models with noise_learn=true")
println("-----------")

println("\nKernel of the GP trained to predict y1 from x=(x1, x2):")
println(emulator_1.machine_learning_tool.models[1].kernel)
println("Learned noise parameter, σ_1:")
learned_σ_1 = sqrt(emulator_1.machine_learning_tool.models[1].kernel.kright.σ2)
println("σ_1 = $learned_σ_1")
# Check if the learned noise is approximately 1
@assert(isapprox(learned_σ_1, 1.0; atol = 0.1))

println("\nKernel of the GP trained to predict y2 from x=(x1, x2):")
println(emulator_1.machine_learning_tool.models[2].kernel)
println("Learned noise parameter, σ_2:")
learned_σ_2 = sqrt(emulator_1.machine_learning_tool.models[2].kernel.kright.σ2)
println("σ_2 = $learned_σ_2")
# Check if the learned noise is approximately 1
@assert(isapprox(learned_σ_2, 1.0; atol = 0.1))
println("------------------------------------------------------------------\n")

# For comparison: When noise_learn is set to false, the observational noise
# is set to 1.0 and is not learned/optimized during the training. But thanks
# to the SVD, 1.0 is the correct value to use.
gauss_proc_2 = GaussianProcess(
    gppackage;
    kernel = nothing, # use default squared exponential kernel
    prediction_type = prediction_type,
    noise_learn = false,
)
emulator_2 = Emulator(gauss_proc_2, iopairs, obs_noise_cov = Σ, normalize_inputs = true)
println("build GP with ", n, " training points")
#optimize the hyperparameters to best fit the GP.
optimize_hyperparameters!(emulator_2)
println("GP trained")

println("\n-----------")
println("Results of training Gaussian Process models with noise_learn=false")
println("-----------")

println("\nKernel of the GP trained to predict y1 from x=(x1, x2):")
# Note: In contrast to the kernels of the gpobj1 models, these ones do not
# have a white noise ("Noise") kernel component
println(emulator_2.machine_learning_tool.models[1].kernel)
# logNoise is given as log(sqrt(noise))
obs_noise_1 = exp(emulator_2.machine_learning_tool.models[1].logNoise.value^2)
println("Observational noise: $obs_noise_1")

println("\nKernel of the GP trained to predict y1 from x=(x1, x2):")
println(emulator_2.machine_learning_tool.models[2].kernel)
# logNoise is given as log(sqrt(noise))
obs_noise_2 = exp(emulator_2.machine_learning_tool.models[2].logNoise.value^2)
println("Observational noise: $obs_noise_2")
println("------------------------------------------------------------------")
