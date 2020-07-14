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

using Plots

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
    "training inputs/parameters; N_samples x input_dim"
    inputs::Array{FT}
    "training data/targets; N_samples x output_dim"
    data::Array{FT}
    "mean of input; 1 x input_dim"
    input_mean::Array{FT}
    "square root of the inverse of the input covariance matrix; input_dim x input_dim"
    sqrt_inv_input_cov::Union{Nothing, Array{FT, 2}}
    "the Gaussian Process (GP) Regression model(s) that are fitted to the given input-data pairs"
    models::Vector
    "whether to fit GP models on normalized inputs ((inputs - input_mean) * sqrt_inv_input_cov)"
    normalized::Bool
    "the singular value decomposition matrix"
    decomposition::SVD
    "the normalization matrix in the svd transformed space"
    sqrt_singular_values::Array{FT,2}
    "prediction type (`y` to predict the data, `f` to predict the latent function)"
    prediction_type::Union{Nothing, PredictionType}
end


"""
    GPObj(inputs, data, package::GPJL; GPkernel::Union{K, KPy, Nothing}=nothing, normalized::Bool=true, svdflag=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x input_dim (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default
                kernel is used. The default kernel is the sum of a Squared
                Exponential kernel and white noise.
"""

function GPObj(
    inputs::Array{FT, 2},
    data::Array{FT, 2},
    truth_cov::Array{FT, 2},
    package::GPJL;
    GPkernel::Union{K, KPy, Nothing}=nothing,
    normalized::Bool=true,
    noise_learn::Bool=true,
    prediction_type::PredictionType=YType()) where {FT<:AbstractFloat, K<:Kernel, KPy<:PyObject}

    # Consistency check
    err = "Number of inputs is not equal to the number of data."
    size(inputs, 1) == size(data, 1) || throw(ArgumentError(err))

    # Initialize vector of GP models
    models = Any[]
    # Number of models (We are fitting one model per output dimension)
    N_models = size(data, 2)
    
    # Make sure inputs and data are arrays of type FT
    inputs = convert(Array{FT}, inputs)
    data = convert(Array{FT}, data)

    # Normalize the inputs if normalized==true 
    input_mean = reshape(mean(inputs, dims=1), 1, :)
    sqrt_inv_input_cov = nothing
    if normalized
        # Normalize and transpose (since the inputs have to be of size 
        # input_dim x N_samples to pass to GPE())
        if ndims(inputs) == 1
            error("Cov not defined for 1d input; can't normalize")
        end
        sqrt_inv_input_cov = convert(Array{FT}, sqrt(inv(cov(inputs, dims=1))))
        GPinputs = convert(Array, ((inputs.-input_mean) * sqrt_inv_input_cov)')
    else
        # Only transpose
        GPinputs = convert(Array, inputs')
    end
    
    # Transform the outputs (transformed_data will come out transposed from
    # this transformation, i.e., with size output_dim x N_samples. This is also 
    # the size that will be required when passing the data to GPE(), so we leave 
    # transformed_data in that size. 
    # Create an SVD decomposition of the covariance:
    # NB: svdfact() may be more efficient / provide positive definiteness information
    # stored as:  svd.U * svd.S *svd.Vt 
    decomposition = svd(truth_cov)
    sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
    transformed_data = sqrt_singular_values_inv * decomposition.Vt * data'

    # Use a default kernel unless a kernel was supplied to GPObj
    if GPkernel==nothing
        println("Using default squared exponential kernel, learning length scale and variance parameters")
        # Construct kernel:
        # Note that the kernels take the signal standard deviations on a
        # log scale as input.
        rbf_len = log.(ones(size(GPinputs, 1)))
        rbf_logstd = log(1.0)
        rbf = SEArd(rbf_len, rbf_logstd)
        kern = rbf
        println("Using default squared exponential kernel: ", kern)
    else
        kern = deepcopy(GPkernel)
        println("Using user-defined kernel", kern)
    end

    if noise_learn
        # Add white noise to kernel
        white_logstd = log(1.0)
        white = Noise(white_logstd)
        kern = kern + white
        println("Learning additive white noise")

        # Make the regularization small. We actually learn 
        # total_noise = white_logstd + logstd_regularization_noise
        magic_number = 1e-3 # magic_number << 1
        logstd_regularization_noise = log(sqrt(magic_number)) 

    else
        # When not learning noise, our SVD transformation implies the 
        # observational noise is the identity.
        logstd_regularization_noise = log(sqrt(1.0)) 
    end
    
    for i in 1:N_models
        # Make a copy of the kernel (because it gets altered in every 
        # iteration)
        println("kernel in GPObj")
        GPkernel_i = deepcopy(kern)
        println(GPkernel_i)
        GPdata_i = transformed_data[i, :]
        # GPE() arguments:
        # GPinputs:    (input_dim x N_samples)
        # GPdata_i:    (N_samples,)
       
        # Zero mean function
        kmean = MeanZero()

        # Instantiate GP model
        m = GPE(GPinputs,
                GPdata_i,
                kmean,
                GPkernel_i, 
                logstd_regularization_noise)
       
        # We choose above to explicitly learn the WhiteKernel as opposed to 
        # using the in built noise=true functionality.
        println("created GP", i)
        optimize!(m, noise=false)
        println("optimized GP", i)
        push!(models, m)
        println(m.kernel)
    end

    return GPObj{FT, typeof(package)}(inputs,
                                      data,
                                      input_mean, 
                                      sqrt_inv_input_cov,
                                      models,
                                      normalized,
                                      decomposition,
                                      inv(sqrt_singular_values_inv),
                                      prediction_type)
end


"""
    GPObj(inputs, data, package::SKLJL, GPkernel::Union{K, KPy, Nothing}=nothing, normalized::Bool=true) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x input_dim (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses or ScikitLearn kernel object. If not supplied, 
                a default kernel is used. The default kernel is the sum of a 
                Squared Exponential kernel and white noise.
"""
function GPObj(
    inputs::Array{FT, 2},
    data::Array{FT, 2},
    truth_cov::Array{FT, 2},
    package::SKLJL;
    GPkernel::Union{K, KPy, Nothing}=nothing,
    normalized::Bool=true,
    noise_learn::Bool=true,
    prediction_type::PredictionType=YType()) where {FT<:AbstractFloat, K<:Kernel, KPy<:PyObject}

    # Consistency check
    err = "Number of inputs is not equal to the number of data."
    size(inputs, 1) == size(data, 1) || throw(ArgumentError(err))

    # Make sure inputs and data are arrays of type FT
    inputs = convert(Array{FT}, inputs)
    data = convert(Array{FT}, data)

    # Initialize vector of GP models
    models = Any[]
    # Number of models (We are fitting one model per output dimension)
    N_models = size(data, 2)

    # Normalize the inputs if normalized==true
    # Note that contrary to the GaussianProcesses.jl (GPJL) GPE, the 
    # ScikitLearn (SKLJL) GaussianProcessRegressor requires inputs to be of 
    # size N_samples x input_dim, so no transposition is needed here
    input_mean = reshape(mean(inputs, dims=1), 1, :)
    sqrt_inv_input_cov = nothing
    if normalized
        if ndims(inputs) == 1
            error("Cov not defined for 1d input; can't normalize")
        end
        sqrt_inv_input_cov = convert(Array{FT}, sqrt(inv(cov(inputs, dims=1))))
        GPinputs = (inputs .- input_mean) * sqrt_inv_input_cov
    else
        GPinputs = inputs
    end

    # Transform the outputs
    # Create an SVD decomposition of the covariance:
    # NB: svdfact() may be more efficient / provide positive definiteness information
    # stored as:  svd.U * svd.S *svd.Vt 
    decomposition = svd(truth_cov)
    sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
    transformed_data = sqrt_singular_values_inv * decomposition.Vt * data'
    
    if GPkernel==nothing
        println("Using default squared exponential kernel, learning length scale and variance parameters")
        # Create default squared exponential kernel
        const_value = 1.0
        var_kern = ConstantKernel(constant_value=const_value,
                                  constant_value_bounds=(1e-05, 10000.0))
        rbf_len = ones(size(GPinputs, 2))
        rbf = RBF(length_scale=rbf_len, length_scale_bounds=(1e-05, 10000.0))
        kern = var_kern * rbf
        println("Using default squared exponential kernel:", kern)
    else
        kern = deepcopy(GPkernel)
        println("Using user-defined kernel",kern)
    end

    if noise_learn
        # Add  white noise to kernel
        white_noise_level = 1.0
        white = WhiteKernel(noise_level=white_noise_level,
                            noise_level_bounds=(1e-05, 10.0))
        kern = kern + white
        println("Learning additive white noise")

        # Make the regularization small. We actually learn 
        # total_noise = white_noise_level + regularization_noise
        magic_number = 1e-3 # magic_number << 1
        regularization_noise = 1e-3
    else
        # When not learning noise, our SVD transformation implies the 
        # observational noise is the identity.
        regularization_noise = 1.0
    end
    
    for i in 1:N_models
        GPkernel_i = deepcopy(kern)
        GPdata_i = transformed_data[i, :]
        m = GaussianProcessRegressor(kernel=GPkernel_i,
                                     n_restarts_optimizer=10,
                                     alpha=regularization_noise)

        println(m.kernel_)
        # ScikitLearn.fit! arguments:
        # GPinputs:    (N_samples x input_dim)
        # GPdata_i:    (N_samples,)
        ScikitLearn.fit!(m, GPinputs, GPdata_i)
        if i==1
            println(m.kernel.hyperparameters)
            print("Completed training of: ")
        end
        print(i,", ")
        push!(models, m)
        println(m.kernel_)
    end
    return GPObj{FT, typeof(package)}(inputs,
                                      data,
                                      input_mean,
                                      sqrt_inv_input_cov,
                                      models,
                                      normalized,
                                      decomposition,
                                      inv(sqrt_singular_values_inv),
                                      YType())
end



"""
    predict(gp::GPObj{FT,GPJL}, new_inputs::Array{FT, 2}) where {FT} = predict(gp, new_inputs, gp.prediction_type)

Evaluate the GP model(s) at new inputs.
  - `gp` - a GPObj
  - `new_inputs` - inputs for which GP model(s) is/are evaluated; N_new_inputs x input_dim

Note: If gp.normalized == true, the new inputs are normalized prior to the prediction
"""
predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}; transform_to_real=false) where {FT} = predict(gp, new_inputs, transform_to_real, gp.prediction_type)

function predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real, ::FType) where {FT}
    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    # new_inputs should be of size N_new_inputs x input_dim, so input_dim
    # has to equal the input dimension of the GP model(s). 
    input_dim = size(gp.inputs, 2)
    output_dim = size(gp.data, 2)
    err = "GP object and input observations do not have consistent dimensions"
    size(new_inputs, 2) == input_dim || throw(ArgumentError(err))

    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end

    M = length(gp.models)
    μσ2 = [predict_f(gp.models[i], new_inputs) for i in 1:M]
    # Return mean(s) and variance(s)
    μ  = reshape(vcat(first.(μσ2)...), size(new_inputs, 1), output_dim)
    σ2 = reshape(vcat(last.(μσ2)...), size(new_inputs, 1), output_dim)

    if transform_to_real
        # Revert back to the original (correlated) coordinate system
        # We created meanvGP = Dinv*Vt*meanv so meanv = V*D*meanvGP
        transformed_μT= gp.decomposition.V * gp.sqrt_singular_values * μ'
        # Get the means back into size N_new_inputs x output_dim
        transformed_μ = convert(Array, transformed_μT')
        transformed_σ2 = zeros(size(σ2))
        # Back transformation of covariance
        for j in 1:size(σ2, 1)
            σ2_j = gp.decomposition.V * gp.sqrt_singular_values * Diagonal(σ2[j, :]) * gp.sqrt_singular_values * gp.decomposition.Vt
            # Extract diagonal elements from σ2_j and add them to σ2 as column 
            # vectors
            # TODO: Should we have an optional input argument `full_cov` - if
            # set to true, the full covariance matrix is returned instead of 
            # only variances (default: false)?
            transformed_σ2[j, :] = diag(σ2_j)
        end
        return transformed_μ, transformed_σ2

    else
        # No back transformation needed
        return μ, σ2
    end
end

function predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real, ::YType) where {FT}
    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    # new_inputs should be of size N_new_inputs x input_dim, so input_dim
    # has to equal the input dimension of the GP model(s). 
    input_dim = size(gp.inputs, 2)
    output_dim = size(gp.data, 2)
    err = "GP object and input observations do not have consistent dimensions"
    size(new_inputs, 2) == input_dim || throw(ArgumentError(err))

    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end

    M = length(gp.models)
    # Predicts columns of inputs so must be transposed to size 
    # input_dim x N_new_inputs
    new_inputs_transp = convert(Array{FT}, new_inputs')
    μσ2 = [predict_y(gp.models[i], new_inputs_transp) for i in 1:M]

    # Return mean(s) and variance(s)
    μ  = reshape(vcat(first.(μσ2)...), size(new_inputs, 1), output_dim)
    σ2 = reshape(vcat(last.(μσ2)...), size(new_inputs, 1), output_dim)

    if transform_to_real
        # Revert back to the original (correlated) coordinate system
        # We created meanvGP = Dinv*Vt*meanv so meanv = V*D*meanvGP
        transformed_μT= gp.decomposition.V * gp.sqrt_singular_values * μ'
        # Get the means back into size N_new_inputs x output_dim
        transformed_μ = convert(Array, transformed_μT')
        transformed_σ2 = zeros(size(σ2))
        # Back transformation of covariance
        for j in 1:size(σ2, 1)
            σ2_j = gp.decomposition.V * gp.sqrt_singular_values * Diagonal(σ2[j, :]) * gp.sqrt_singular_values * gp.decomposition.Vt
            # Extract diagonal elements from σ2_j and add them to σ2 as column 
            # vectors
            # TODO: Should we have an optional input argument `full_cov` - if
            # set to true, the full covariance matrix is returned instead of 
            # only variances (default: false)?
            transformed_σ2[j, :] = diag(σ2_j)
        end
        return transformed_μ, transformed_σ2

    else
        # No back transformation needed
        return μ, σ2
    end

end

function predict(gp::GPObj{FT, SKLJL}, new_inputs::Array{FT, 2}, transform_to_real) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    # new_inputs should be of size N_new_inputs x input_dim, so input_dim
    # has to equal the input dimension of the GP model(s). 
    input_dim = size(gp.inputs, 2)
    output_dim = size(gp.data, 2)
    err = "GP object and input observations do not have consistent dimensions"
    size(new_inputs, 2) == input_dim || throw(ArgumentError(err))

    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end

    M = length(gp.models)
    # Predicts rows of inputs; no need to transpose
    μσ = [gp.models[i].predict(new_inputs, return_std=true) for i in 1:M]
    # Return mean(s) and standard deviations(s)
    μ = reshape(vcat(first.(μσ)...), size(new_inputs)[1], output_dim)
    σ = reshape(vcat(last.(μσ)...), size(new_inputs)[1], output_dim)
    σ2 = σ*σ
    
    if transform_to_real
        # Revert back to the original (correlated) coordinate system
        # We created meanvGP = Dinv*Vt*meanv so meanv = V*D*meanvGP
        transformed_μT= gp.decomposition.V * gp.sqrt_singular_values * μ'
        # Get the means back into size N_new_inputs x output_dim
        transformed_μ = convert(Array, transformed_μT')
        println("size transformed μ")
        println(size(transformed_μ))
        transformed_σ2 = zeros(size(σ2))
        # Back transformation of covariance
        for j in 1:size(σ2, 1)
            σ2_j = gp.decomposition.V * gp.sqrt_singular_values * Diagonal(σ2[j, :]) * gp.sqrt_singular_values * gp.decomposition.Vt
            # Extract diagonal elements from σ2_j and add them to σ2 as column 
            # vectors
            # TODO: Should we have an optional input argument `full_cov` - if
            # set to true, the full covariance matrix is returned instead of 
            # only variances (default: false)?
            transformed_σ2[j, :] = diag(σ2_j)
        end
        return transformed_μ, transformed_σ2

    else
        # No back transformation needed
        return μ, σ2
    end

end

end
