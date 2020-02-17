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
struct GPObj{FT<:AbstractFloat,GPM}
    "training points/parameters; N_parameters x N_samples"
    inputs::Array{FT, 2}
    "output data (`outputs = G(inputs)`); N_parameters x N_samples"
    data::Array{FT, 2}
    "the Gaussian Process Regression model(s) that are fitted to the given input-output pairs"
    models::Vector
    "Data prediction type"
    prediction_type::Union{Nothing,PredictionType}
end

"""
    GPObj(inputs, data, package::GPJL, GPkernel::Union{K, KPy, Nothing}=nothing; prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x N_parameters (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default
                kernel is used. The default kernel is the sum of a Squared
                Exponential kernel and white noise.
"""
function GPObj(inputs, data, package::GPJL, GPkernel::Union{K, KPy, Nothing}=nothing; prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}
    FT = eltype(data)
    models = Any[]
    outputs = convert(Array{FT}, data')
    inputs = convert(Array{FT}, inputs')

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

    for i in 1:size(outputs, 1)
        # Make a copy of GPkernel (because the kernel gets altered in
        # every iteration)
        GPkernel_i = deepcopy(GPkernel)
        # inputs: N_param x N_samples
        # outputs: N_data x N_samples
        logstd_obs_noise = log(sqrt(0.5)) # log standard dev of obs noise
        # Zero mean function
        kmean = MeanZero()
        m = GPE(inputs, outputs[i, :], kmean, GPkernel_i, logstd_obs_noise)
        optimize!(m, noise=false)
        push!(models, m)
    end
    return GPObj{FT, typeof(package)}(inputs, outputs, models, prediction_type)
end

"""
    GPObj(inputs, data, package::SKLJL, GPkernel::Union{K, KPy, Nothing}=nothing) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x N_parameters (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default
                kernel is used. The default kernel is the sum of a Squared
                Exponential kernel and white noise.
"""
function GPObj(inputs, data, package::SKLJL, GPkernel::Union{K, KPy, Nothing}=nothing) where {K<:Kernel, KPy<:PyObject}
    FT = eltype(data)
    models = Any[]
    outputs = convert(Array{FT}, data')
    inputs = convert(Array{FT}, inputs)
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
    for i in 1:size(outputs,1)
        out = reshape(outputs[i,:], (size(outputs, 2), 1))
        m = GaussianProcessRegressor(kernel=GPkernel,
                                     n_restarts_optimizer=10,
                                     alpha=0.0, normalize_y=true)
        ScikitLearn.fit!(m, inputs, out)
        if i==1
            println(m.kernel.hyperparameters)
            print("Completed training of: ")
        end
        print(i,", ")
        push!(models, m)
    end
    return GPObj{FT, typeof(package)}(inputs, outputs, models, nothing)
end


predict(gp::GPObj{FT,GPJL}, new_inputs::Array{FT}) where {FT} = predict(gp, new_inputs, gp.prediction_type)

function predict(gp::GPObj{FT,GPJL},
                 new_inputs::Array{FT},
                 ::FType) where {FT}
    M = length(gp.models)
    μσ2 = [predict_f(gp.models[i], new_inputs) for i in 1:M]
    return first.(μσ2), last.(μσ2)
end
function predict(gp::GPObj{FT,GPJL},
                 new_inputs::Array{FT},
                 ::YType) where {FT}
    M = length(gp.models)
    # predicts columns of inputs so must be transposed
    new_inputs = convert(Array{FT}, new_inputs')
    μσ2 = [predict_y(gp.models[i], new_inputs) for i in 1:M]
    return first.(μσ2), last.(μσ2)
end

function predict(gp::GPObj{FT,SKLJL},
                 new_inputs::Array{FT}) where {FT}
    M = length(gp.models)
    μσ = [gp.models[i].predict(new_inputs, return_std=true) for i in 1:M]
    mean = first.(μσ)
    return first.(μσ), last.(μσ).^2
end

end # module
