
import GaussianProcesses: predict
using GaussianProcesses

using EnsembleKalmanProcesses.DataContainers

using DocStringExtensions
using PyCall
using ScikitLearn
using AbstractGPs
using KernelFunctions
const pykernels = PyNULL()
const pyGP = PyNULL()
function __init__()
    copy!(pykernels, pyimport_conda("sklearn.gaussian_process.kernels", "scikit-learn=1.1.1"))
    copy!(pyGP, pyimport_conda("sklearn.gaussian_process", "scikit-learn=1.1.1"))
end

#exports (from Emulator)
export GaussianProcess

export GPJL, SKLJL
export YType, FType

"""
$(DocStringExtensions.TYPEDEF)

Type to dispatch which GP package to use:

 - `GPJL` for GaussianProcesses.jl,
 - `SKLJL` for the ScikitLearn GaussianProcessRegressor.
"""
abstract type GaussianProcessesPackage end
struct GPJL <: GaussianProcessesPackage end
struct SKLJL <: GaussianProcessesPackage end

"""
$(DocStringExtensions.TYPEDEF)

Predict type for `GPJL` in GaussianProcesses.jl:
 - `YType`
 - `FType` latent function.
"""
abstract type PredictionType end
struct YType <: PredictionType end
struct FType <: PredictionType end

"""
$(DocStringExtensions.TYPEDEF)

Structure holding training input and the fitted Gaussian process regression
models.

# Fields
$(DocStringExtensions.TYPEDFIELDS)

"""
struct GaussianProcess{GPPackage, FT} <: MachineLearningTool
    "The Gaussian Process (GP) Regression model(s) that are fitted to the given input-data pairs."
    models::Vector{Union{<:GaussianProcesses.GPE, <:PyObject, Nothing}}
    "Kernel object."
    kernel::Union{<:Kernel, <:PyObject, Nothing}
    "Learn the noise with the White Noise kernel explicitly?"
    noise_learn::Bool
    "Additional observational or regularization noise in used in GP algorithms"
    alg_reg_noise::FT
    "Prediction type (`y` to predict the data, `f` to predict the latent function)."
    prediction_type::PredictionType

end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

 - `package` - GaussianProcessPackage object.
 - `kernel` - GaussianProcesses kernel object. Default is a Squared Exponential kernel.
 - `noise_learn` - Boolean to additionally learn white noise in decorrelated space. Default is true.
 - `alg_reg_noise` - Float to fix the (small) regularization parameter of algorithms when `noise_learn = true`
 - `prediction_type` - PredictionType object. Default predicts data, not latent function (FType()).
"""
function GaussianProcess(
    package::GPPkg;
    kernel::Union{K, KPy, Nothing} = nothing,
    noise_learn = true,
    alg_reg_noise::FT = 1e-3,
    prediction_type::PredictionType = YType(),
) where {GPPkg <: GaussianProcessesPackage, K <: Kernel, KPy <: PyObject, FT <: AbstractFloat}

    # Initialize vector for GP models
    models = Vector{Union{<:GaussianProcesses.GPE, <:PyObject, Nothing}}(undef, 0)

    # the algorithm regularization noise is set to some small value if we are learning noise, else
    # it is fixed to the correct value (1.0)
    if !(noise_learn)
        alg_reg_noise = 1.0
    end

    return GaussianProcess{typeof(package), FT}(models, kernel, noise_learn, alg_reg_noise, prediction_type)
end

# First we create  the GPJL implementation
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Method to build Gaussian process models based on the package.
"""
function build_models!(
    gp::GaussianProcess{GPJL},
    input_output_pairs::PairedDataContainer{FT},
) where {FT <: AbstractFloat}
    # get inputs and outputs
    input_values = get_inputs(input_output_pairs)
    output_values = get_outputs(input_output_pairs)

    # Number of models (We are fitting one model per output dimension, as data is decorrelated)
    models = gp.models
    if length(gp.models) > 0 # check to see if gp already contains models
        @warn "GaussianProcess already built. skipping..."
        return
    end
    N_models = size(output_values, 1) #size(transformed_data)[1]


    # Use a default kernel unless a kernel was supplied to GaussianProcess
    if gp.kernel === nothing
        println("Using default squared exponential kernel, learning length scale and variance parameters")
        # Construct kernel:
        # Note that the kernels take the signal standard deviations on a
        # log scale as input.
        rbf_len = log.(ones(size(input_values, 1)))
        rbf_logstd = log(1.0)
        rbf = SEArd(rbf_len, rbf_logstd)
        kern = rbf
        println("Using default squared exponential kernel: ", kern)
    else
        kern = deepcopy(gp.kernel)
        println("Using user-defined kernel", kern)
    end

    if gp.noise_learn
        # Add white noise to kernel
        white_logstd = log(1.0)
        white = Noise(white_logstd)
        kern = kern + white
        println("Learning additive white noise")
    end
    logstd_regularization_noise = log(sqrt(gp.alg_reg_noise))

    for i in 1:N_models
        # Make a copy of the kernel (because it gets altered in every
        # iteration)
        kernel_i = deepcopy(kern)
        println("kernel in GaussianProcess:")
        println(kernel_i)
        data_i = output_values[i, :]
        # GaussianProcesses.GPE() arguments:
        # input_values:    (input_dim × N_samples)
        # GPdata_i:    (N_samples,)

        # Zero mean function
        kmean = MeanZero()

        # Instantiate GP model
        m = GaussianProcesses.GPE(input_values, output_values[i, :], kmean, kernel_i, logstd_regularization_noise)

        println("created GP: ", i)
        push!(models, m)

    end

end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimize Gaussian process hyperparameters using in-build package method.

Warning: if one uses `GPJL()` and wishes to modify positional arguments. The first positional argument must be the `Optim` method (default `LBGFS()`).
"""
function optimize_hyperparameters!(gp::GaussianProcess{GPJL}, args...; kwargs...)
    N_models = length(gp.models)
    for i in 1:N_models
        # always regress with noise_learn=false; if gp was created with noise_learn=true
        # we've already explicitly added noise to the kernel

        optimize!(gp.models[i], args...; noise = false, kwargs...)
        println("optimized hyperparameters of GP: ", i)
        println(gp.models[i].kernel)
    end
end

# subroutine with common predict() logic
function _predict(
    gp::GaussianProcess,
    new_inputs::AbstractMatrix{FT},
    predict_method::Function,
) where {FT <: AbstractFloat}
    M = length(gp.models)
    N_samples = size(new_inputs, 2)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(M, N_samples)
    σ2 = zeros(M, N_samples)
    for i in 1:M
        μ[i, :], σ2[i, :] = predict_method(gp.models[i], new_inputs)
    end

    return μ, σ2
end

predict(gp::GaussianProcess{GPJL}, new_inputs::AbstractMatrix{FT}, ::YType) where {FT <: AbstractFloat} =
    _predict(gp, new_inputs, GaussianProcesses.predict_y)

predict(gp::GaussianProcess{GPJL}, new_inputs::AbstractMatrix{FT}, ::FType) where {FT <: AbstractFloat} =
    _predict(gp, new_inputs, GaussianProcesses.predict_f)

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Predict means and covariances in decorrelated output space using Gaussian process models.
"""
predict(gp::GaussianProcess{GPJL}, new_inputs::AbstractMatrix{FT}) where {FT <: AbstractFloat} =
    predict(gp, new_inputs, gp.prediction_type)


#now we build the SKLJL implementation
function build_models!(
    gp::GaussianProcess{SKLJL},
    input_output_pairs::PairedDataContainer{FT},
) where {FT <: AbstractFloat}
    # get inputs and outputs
    input_values = permutedims(get_inputs(input_output_pairs), (2, 1))
    output_values = get_outputs(input_output_pairs)

    # Number of models (We are fitting one model per output dimension, as data is decorrelated)
    models = gp.models
    if length(gp.models) > 0 # check to see if gp already contains models
        @warn "GaussianProcess already built. skipping..."
        return
    end

    N_models = size(output_values, 1) #size(transformed_data)[1]

    if gp.kernel === nothing
        println("Using default squared exponential kernel, learning length scale and variance parameters")
        # Create default squared exponential kernel
        const_value = 1.0
        var_kern = pykernels.ConstantKernel(constant_value = const_value, constant_value_bounds = (1e-5, 1e4))
        rbf_len = ones(size(input_values, 2))
        rbf = pykernels.RBF(length_scale = rbf_len, length_scale_bounds = (1e-5, 1e4))
        kern = var_kern * rbf
        println("Using default squared exponential kernel:", kern)
    else
        kern = deepcopy(gp.kernel)
        println("Using user-defined kernel", kern)
    end

    if gp.noise_learn
        # Add white noise to kernel
        white_noise_level = 1.0
        white = pykernels.WhiteKernel(noise_level = white_noise_level, noise_level_bounds = (1e-05, 10.0))
        kern = kern + white
        println("Learning additive white noise")
    end

    regularization_noise = gp.alg_reg_noise

    for i in 1:N_models
        kernel_i = deepcopy(kern)
        data_i = output_values[i, :]
        m = pyGP.GaussianProcessRegressor(kernel = kernel_i, n_restarts_optimizer = 10, alpha = regularization_noise)

        # ScikitLearn.fit! arguments:
        # input_values:    (N_samples × input_dim)
        # data_i:    (N_samples,)
        ScikitLearn.fit!(m, input_values, data_i)
        if i == 1
            println(m.kernel.hyperparameters)
            print("Completed training of: ")
        end
        println(i, ", ")
        push!(models, m)
        println(m.kernel)
    end
end


function optimize_hyperparameters!(gp::GaussianProcess{SKLJL}, args...; kwargs...)
    println("SKlearn, already trained. continuing...")
end

function _SKJL_predict_function(gp_model::PyObject, new_inputs::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    # SKJL based on rows not columns; need to transpose inputs
    μ, σ = gp_model.predict(new_inputs', return_std = true)
    return μ, (σ .* σ)
end
function predict(gp::GaussianProcess{SKLJL}, new_inputs::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    μ, σ2 = _predict(gp, new_inputs, _SKJL_predict_function)

    # for SKLJL does not return the observational noise (even if return_std = true)
    # we must add contribution depending on whether we learnt the noise or not.
    σ2[:, :] = σ2[:, :] .+ gp.alg_reg_noise

    return μ, σ2
end
