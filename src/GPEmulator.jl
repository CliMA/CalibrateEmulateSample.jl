module GPEmulator

using Statistics
using Distributions
using LinearAlgebra
using GaussianProcesses
using ScikitLearn
using Optim
using DocStringExtensions

using PyCall
@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)

export GPObj
export predict

export GPJL, SKLJL
export YType, FType

"""
    GaussianProcessesPackage

Type to dispatch which GP package to use

 - `GPJL` for GaussianProcesses.jl,
 - `SKLJL` for the ScikitLearn GaussianProcessRegressor
"""
abstract type GaussianProcessesPackage end
struct GPJL <: GaussianProcessesPackage end
struct SKLJL <: GaussianProcessesPackage end

"""
    PredictionType

Predict type for `GPJL` in GaussianProcesses.jl
 - `YType`
 - `FType` latent function
"""
abstract type PredictionType end
struct YType <: PredictionType end
struct FType <: PredictionType end

"""
    GPObj{FT<:AbstractFloat}

Structure holding training input and the fitted Gaussian Process Regression
models.

# Fields
$(DocStringExtensions.FIELDS)

"""
struct GPObj{FT<:AbstractFloat, GPM}
    "training inputs/parameters; N_samples x N_parameters"
    inputs::Array{FT}
    "training data/targets; N_samples x N_data"
    data::Array{FT}
    "mean of input; 1 x N_parameters"
    input_mean::Array{FT}
    "square root of the inverse of the input covariance matrix; N_parameters x N_parameters"
    sqrt_inv_input_cov::Union{Nothing, Array{FT, 2}}
    "the Gaussian Process (GP) Regression model(s) that are fitted to the given input-data pairs"
    models::Vector
    "whether to fit GP models on normalized inputs ((inputs - input_mean) * sqrt_inv_input_cov)"
    normalized::Bool
    "prediction type (`y` to predict the data, `f` to predict the latent function)"
    prediction_type::Union{Nothing, PredictionType}
end


"""
    GPObj(inputs, data, package::GPJL; GPkernel::Union{K, KPy, Nothing}=nothing, normalized::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x N_parameters (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default
                kernel is used. The default kernel is the sum of a Squared
                Exponential kernel and white noise.
"""
function GPObj(inputs, data, package::GPJL; GPkernel::Union{K, KPy, Nothing}=nothing, normalized::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}
    FT = eltype(data)
    models = Any[]

    # Make sure inputs and data are arrays of type FT
    inputs = convert(Array{FT}, inputs)
    data = convert(Array{FT}, data)

    input_mean = reshape(mean(inputs, dims=1), 1, :)
    sqrt_inv_input_cov = nothing
    if normalized
        if ndims(inputs) == 1
            error("Cov not defined for 1d input; can't normalize")
        end
        sqrt_inv_input_cov = convert(Array{FT}, sqrt(inv(cov(inputs, dims=1))))
        inputs = (inputs .- input_mean) * sqrt_inv_input_cov
    end
        
    # Use a default kernel unless a kernel was supplied to GPObj
    if GPkernel==nothing
        # Construct kernel:
        # Note that the kernels take the signal standard deviations on a
        # log scale as input.
        rbf_len = log.(ones(size(inputs, 2)))
        rbf_logstd = log(1.0)
        rbf = SEArd(rbf_len, rbf_logstd)
        # regularize with white noise
        white_logstd = log(1.0)
        white = Noise(white_logstd)
        # construct kernel
        GPkernel = rbf + white
    end

    for i in 1:size(data, 2)
        # Make a copy of GPkernel (because the kernel gets altered in
        # every iteration)
        GPkernel_i = deepcopy(GPkernel)
        # inputs: N_samples x N_parameters
        # data: N_samples x N_data
        logstd_obs_noise = log(sqrt(0.5)) # log standard dev of obs noise
        # Zero mean function
        kmean = MeanZero()
        m = GPE(inputs', dropdims(data[:, i]', dims=1), kmean, GPkernel_i, 
                logstd_obs_noise)
        optimize!(m, noise=false)
        push!(models, m)
    end
    return GPObj{FT, typeof(package)}(inputs, data, input_mean, 
                                      sqrt_inv_input_cov, models, normalized,
                                      prediction_type)
end


"""
    GPObj(inputs, data, package::SKLJL, GPkernel::Union{K, KPy, Nothing}=nothing, normalized::Bool=true) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x N_parameters (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses or ScikitLearn kernel object. If not supplied, 
                a default kernel is used. The default kernel is the sum of a 
                Squared Exponential kernel and white noise.
"""
function GPObj(inputs, data, package::SKLJL; GPkernel::Union{K, KPy, Nothing}=nothing, normalized::Bool=true) where {K<:Kernel, KPy<:PyObject}
    FT = eltype(data)
    models = Any[]

    # Make sure inputs and data are arrays of type FT
    inputs = convert(Array{FT}, inputs)
    data = convert(Array{FT}, data)

    input_mean = reshape(mean(inputs, dims=1), 1, :)
    sqrt_inv_input_cov = nothing
    if normalized
        if ndims(inputs) == 1
            error("Cov not defined for 1d input; can't normalize")
        end
        sqrt_inv_input_cov = convert(Array{FT}, sqrt(inv(cov(inputs, dims=1))))
        inputs = (inputs .- input_mean) * sqrt_inv_input_cov
    end

    if GPkernel==nothing
        const_value = 1.0
        var_kern = ConstantKernel(constant_value=const_value,
                                 constant_value_bounds=(1e-05, 10000.0))
        rbf_len = ones(size(inputs, 2))
        rbf = RBF(length_scale=rbf_len, length_scale_bounds=(1e-05, 10000.0))
        white_noise_level = 1.0
        white = WhiteKernel(noise_level=white_noise_level,
                            noise_level_bounds=(1e-05, 10.0))
        GPkernel = var_kern * rbf + white
    end
    for i in 1:size(data, 2)
        m = GaussianProcessRegressor(kernel=GPkernel,
                                     n_restarts_optimizer=10,
                                     alpha=0.0, normalize_y=true)
        ScikitLearn.fit!(m, inputs, data[:, i])
        if i==1
            println(m.kernel.hyperparameters)
            print("Completed training of: ")
        end
        print(i,", ")
        push!(models, m)
    end
    return GPObj{FT, typeof(package)}(inputs, data, input_mean, sqrt_inv_input_cov, 
                                      models, normalized, YType())
end


"""
  predict(gp::GPObj{FT,GPJL}, new_inputs::Array{FT}) where {FT} = predict(gp, new_inputs, gp.prediction_type)

Evaluate the GP model(s) at new inputs.
  - `gp` - a GPObj
  - `new_inputs` - inputs for which GP model(s) is/are evaluated; N_new_inputs x N_parameters

Note: If gp.normalized == true, the new inputs are normalized prior to the prediction
"""
predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT}) where {FT} = predict(gp, new_inputs, gp.prediction_type)

function predict(gp::GPObj{FT, GPJL},
                 new_inputs::Array{FT},
                 ::FType) where {FT}
    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end
    M = length(gp.models)
    μσ2 = [predict_f(gp.models[i], new_inputs) for i in 1:M]
    # Return mean(s) and standard deviation(s)
    return vcat(first.(μσ2)...), vcat(last.(μσ2)...)
end

function predict(gp::GPObj{FT, GPJL},
                 new_inputs::Array{FT},
                 ::YType) where {FT}
    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end
    M = length(gp.models)
    # predicts columns of inputs so must be transposed
    new_inputs = convert(Array{FT}, new_inputs')
    μσ2 = [predict_y(gp.models[i], new_inputs) for i in 1:M]
    # Return mean(s) and standard deviation(s)
    return vcat(first.(μσ2)...), vcat(last.(μσ2)...)
end

function predict(gp::GPObj{FT, SKLJL},
                 new_inputs::Array{FT}) where {FT}
    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end
    M = length(gp.models)
    μσ = [gp.models[i].predict(new_inputs, return_std=true) for i in 1:M]
    # Return mean(s) and standard deviation(s)
    return vcat(first.(μσ)...), vcat(last.(μσ)...).^2
end

end # module
