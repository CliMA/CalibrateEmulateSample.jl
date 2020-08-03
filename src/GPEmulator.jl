module GPEmulator

using Statistics
using Distributions
using LinearAlgebra
using GaussianProcesses
using DocStringExtensions

using PyCall
using ScikitLearn
const pykernels = PyNULL()
const pyGP = PyNULL()
function __init__()
    copy!(pykernels, pyimport("sklearn.gaussian_process.kernels"))
    copy!(pyGP, pyimport("sklearn.gaussian_process"))
end

export GPObj
export predict
export gp_sqbias

export GPJL, SKLJL
export YType, FType
export svd_transform, svd_reverse_transform_mean_cov

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
    "the Gaussian Process (GP) Regression model(s) that are fitted to the given input-data pairs"
    models::Vector
    "the singular value decomposition of obs_noise_cov, such that obs_noise_cov = decomposition.U * Diagonal(decomposition.S) * decomposition.Vt."
    decomposition::Union{SVD, Nothing}
    "whether to fit GP models on normalized inputs ((inputs - input_mean) * sqrt_inv_input_cov)"
    normalized::Bool
    "square root of the inverse of the input covariance matrix; input_dim x input_dim"
    sqrt_inv_input_cov::Union{Nothing, Array{FT, 2}}
    "prediction type (`y` to predict the data, `f` to predict the latent function)"
    prediction_type::Union{Nothing, PredictionType}
end


"""
GPObj(inputs::Array{FT, 2}, data::Array{FT, 2}, package::GPJL; GPkernel::Union{K, KPy, Nothing}=nothing, obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing, normalized::Bool=true, noise_learn::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Inputs and data of size N_samples x input_dim (both arrays will be transposed in the construction of the GPObj)

 - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default Squared Exponential kernel is used.
"""

function GPObj(
    inputs::Array{FT, 2},
    data::Array{FT, 2},
    package::GPJL;
    GPkernel::Union{K, KPy, Nothing}=nothing,
    obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing,
    normalized::Bool=true,
    noise_learn::Bool=true,
    non_gaussian::Bool=false,
    prediction_type::PredictionType=YType()) where {FT<:AbstractFloat, K<:Kernel, KPy<:PyObject}

    # Consistency checks
    input_dim = size(inputs, 2)
    output_dim = size(data, 2)
    err1 = "Number of inputs is not equal to the number of data."
    size(inputs, 1) == size(data, 1) || throw(ArgumentError(err1))
    if obs_noise_cov != nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end

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
        sqrt_inv_input_cov = convert(Array{FT}, sqrt(inv(cov(inputs, dims=1))))
        GPinputs = convert(Array, ((inputs.-input_mean) * sqrt_inv_input_cov)')
    else
        # Only transpose
        GPinputs = convert(Array, inputs')
    end
    
    # Transform data if obs_noise_cov available (if obs_noise_cov==nothing, 
    # transformed_data is equal to data)
    transformed_data, decomposition = svd_transform(data, obs_noise_cov)

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
        magic_number = 1e-6 # magic_number << 1
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
        GPdata_i = transformed_data[:, i]
        # GPE() arguments:
        # GPinputs:    (input_dim x N_samples)
        # GPdata_i:    (N_samples,)
       
        # Zero mean function
        kmean = MeanZero()

        # Instantiate GP model
        if non_gaussian
            l = StuTLik(50, 0.1)
            m = GPA(GPinputs,
                    GPdata_i,
                    kmean,
                    GPkernel_i, 
                    l)
            println("created GPA", i)
            set_priors!(m.lik,[Normal(1.0,1.0)])
            set_priors!(m.kernel,[Normal(0.0,2.0) for i in 1:size(GPinputs, 1)+1])
            samples = mcmc(m;nIter=10000,burn=1000,thin=2);
            for i in 1:size(samples,2)
                GaussianProcesses.set_params!(m,samples[:,i])
                update_target!(m)
                # PREDICTION GOES HERE
            end
            println("optimized GPA", i)
        else
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
        end
        push!(models, m)
        println(m.kernel)
    end

    return GPObj{FT, typeof(package)}(inputs,
                                      data,
                                      input_mean, 
                                      models,
                                      decomposition,
                                      normalized,
                                      sqrt_inv_input_cov,
                                      prediction_type)
end


"""
GPObj(inputs::Array{FT, 2}, data::Array{FT, 2}, package::SKLJL; GPkernel::Union{K, KPy, Nothing}=nothing, obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing, normalized::Bool=true, noise_learn::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}


Inputs and data of size N_samples x input_dim

 - `GPkernel` - ScikitLearn kernel object. If not supplied, a default Squared Exponential kernel is used.
"""
function GPObj(
    inputs::Array{FT, 2},
    data::Array{FT, 2},
    package::SKLJL;
    GPkernel::Union{K, KPy, Nothing}=nothing,
    obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing,
    normalized::Bool=true,
    noise_learn::Bool=true,
    prediction_type::PredictionType=YType()) where {FT<:AbstractFloat, K<:Kernel, KPy<:PyObject}

    # Consistency checks
    input_dim = size(inputs, 2)
    output_dim = size(data, 2)
    err1 = "Number of inputs is not equal to the number of data."
    size(inputs, 1) == size(data, 1) || throw(ArgumentError(err1))
    if obs_noise_cov != nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end

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
        sqrt_inv_input_cov = convert(Array{FT}, sqrt(inv(cov(inputs, dims=1))))
        GPinputs = (inputs .- input_mean) * sqrt_inv_input_cov
    else
        GPinputs = inputs
    end

    # Transform data if obs_noise_cov available (if obs_noise_cov==nothing, 
    # transformed_data is equal to data)
    transformed_data, decomposition = svd_transform(data, obs_noise_cov)
    if GPkernel==nothing
        println("Using default squared exponential kernel, learning length scale and variance parameters")
        # Create default squared exponential kernel
        const_value = 1.0
        var_kern = pykernels.ConstantKernel(constant_value=const_value,
                                            constant_value_bounds=(1e-5, 1e4))
        rbf_len = ones(size(GPinputs, 2))
        rbf = pykernels.RBF(length_scale=rbf_len, 
                            length_scale_bounds=(1e-5, 1e4))
        kern = var_kern * rbf
        println("Using default squared exponential kernel:", kern)
    else
        kern = deepcopy(GPkernel)
        println("Using user-defined kernel", kern)
    end

    if noise_learn
        # Add white noise to kernel
        white_noise_level = 1.0
        white = pykernels.WhiteKernel(noise_level=white_noise_level,
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
        GPdata_i = transformed_data[:, i]
        m = pyGP.GaussianProcessRegressor(kernel=GPkernel_i,
                                          n_restarts_optimizer=10,
                                          alpha=regularization_noise)

        # ScikitLearn.fit! arguments:
        # GPinputs:    (N_samples x input_dim)
        # GPdata_i:    (N_samples,)
        ScikitLearn.fit!(m, GPinputs, GPdata_i)
        if i==1
            println(m.kernel.hyperparameters)
            print("Completed training of: ")
        end
        println(i, ", ")
        push!(models, m)
        println(m.kernel)
    end
    return GPObj{FT, typeof(package)}(inputs,
                                      data,
                                      input_mean,
                                      models,
                                      decomposition,
                                      normalized,
                                      sqrt_inv_input_cov,
                                      YType())
end


"""
    predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real::Bool=false) where {FT} = predict(gp, new_inputs, gp.prediction_type)

Evaluate the GP model(s) at new inputs.
  - `gp` - a GPObj
  - `new_inputs` - inputs for which GP model(s) is/are evaluated; N_new_inputs x input_dim

Returns the predicted mean(s) and covariance(s) at the input points. 
Means: matrix of size N_new_inputs x output_dim
Covariances: vector of length N_new_inputs, each element is a matrix of size output_dim x output_dim. If the output is 1-dimensional, a N_new_inputs x 1 array of scalar variances is returned rather than a vector of 1x1 matrices.

Note: If gp.normalized == true, the new inputs are normalized prior to the prediction
"""
predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}; transform_to_real::Bool=false) where {FT} = predict(gp, new_inputs, transform_to_real, gp.prediction_type)

function predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real, ::FType) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    # new_inputs should be of size N_new_inputs x input_dim, so input_dim
    # has to equal the input dimension of the GP model(s). 
    input_dim = size(gp.inputs, 2)
    output_dim = size(gp.data, 2)
    N_new_inputs = size(new_inputs, 1)
    err = "GP object and input observations do not have consistent dimensions"
    size(new_inputs, 2) == input_dim || throw(ArgumentError(err))

    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end

    M = length(gp.models)
    # Predicts columns of inputs so must be transposed to size 
    # input_dim x N_new_inputs
    new_inputs_transp = convert(Array{FT}, new_inputs')
    μσ2 = [predict_f(gp.models[i], new_inputs_transp) for i in 1:M]

    # Return mean(s) and variance(s)
    μ  = reshape(vcat(first.(μσ2)...), N_new_inputs, output_dim)
    σ2 = reshape(vcat(last.(μσ2)...), N_new_inputs, output_dim)

    if transform_to_real && gp.decomposition != nothing
        μ_pred, σ2_pred = svd_reverse_transform_mean_cov(μ, σ2,
                                                         gp.decomposition)
    elseif transform_to_real && gp.decomposition == nothing
        err = """Need SVD decomposition to transform back to original space, 
                 but GPObj.decomposition == nothing. 
                 Try setting transform_to_real=false"""
        throw(ArgumentError(err))
    else
        μ_pred = μ
        # Convert to vector of (diagonal) matrices to match the format of 
        # σ2_pred when transform_to_real=true
        σ2_pred = vec([Diagonal(σ2[j, :]) for j in 1:N_new_inputs])
    end

    if output_dim == 1
        σ2_pred = reshape([σ2_pred[i][1] for i in 1:N_new_inputs], N_new_inputs, 1)
    end

    return μ_pred, σ2_pred
end

function predict(gp::GPObj{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real, ::YType) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    # new_inputs should be of size N_new_inputs x input_dim, so input_dim
    # has to equal the input dimension of the GP model(s). 
    input_dim = size(gp.inputs, 2)
    output_dim = size(gp.data, 2)
    N_new_inputs = size(new_inputs, 1)
    err = "GP object and input observations do not have consistent dimensions"
    size(new_inputs, 2) == input_dim || throw(ArgumentError(err))

    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end

    M = length(gp.models)
    # Predicts columns of inputs so must be transposed to size 
    # input_dim x N_new_inputs
    new_inputs_transp = convert(Array{FT}, new_inputs')
    μσ2 = [predict_f(gp.models[i], new_inputs_transp) for i in 1:M]

    # Return mean(s) and variance(s)
    μ  = reshape(vcat(first.(μσ2)...), N_new_inputs, output_dim)
    σ2 = reshape(vcat(last.(μσ2)...), N_new_inputs, output_dim)

    if transform_to_real && gp.decomposition != nothing
        μ_pred, σ2_pred = svd_reverse_transform_mean_cov(μ, σ2, 
                                                         gp.decomposition)
    elseif transform_to_real && gp.decomposition == nothing
        err = """Need SVD decomposition to transform back to original space, 
                 but GPObj.decomposition == nothing. 
                 Try setting transform_to_real=false"""
        throw(ArgumentError(err))
    else
        μ_pred = μ
        # Convert to vector of (diagonal) matrices to match the format of 
        # σ2_pred when transform_to_real=true
        σ2_pred = vec([Diagonal(σ2[j, :]) for j in 1:N_new_inputs])
    end

    if output_dim == 1
        σ2_pred = reshape([σ2_pred[i][1] for i in 1:N_new_inputs], N_new_inputs, 1)
    end

    return μ_pred, σ2_pred
end

function predict(gp::GPObj{FT, SKLJL}, new_inputs::Array{FT, 2}, transform_to_real::Bool=false) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    # new_inputs should be of size N_new_inputs x input_dim, so input_dim
    # has to equal the input dimension of the GP model(s). 
    input_dim = size(gp.inputs, 2)
    output_dim = size(gp.data, 2)
    N_new_inputs = size(new_inputs, 1)
    err = "GP object and input observations do not have consistent dimensions"
    size(new_inputs, 2) == input_dim || throw(ArgumentError(err))

    if gp.normalized
        new_inputs = (new_inputs .- gp.input_mean) * gp.sqrt_inv_input_cov
    end

    M = length(gp.models)
    # Predicts rows of inputs; no need to transpose
    μσ = [gp.models[i].predict(new_inputs, return_std=true) for i in 1:M]
    # Return mean(s) and standard deviations(s)
    μ = reshape(vcat(first.(μσ)...), N_new_inputs, output_dim)
    σ = reshape(vcat(last.(μσ)...), N_new_inputs, output_dim)
    σ2 = σ .* σ

    if transform_to_real && gp.decomposition != nothing
        μ_pred, σ2_pred = svd_reverse_transform_mean_cov(μ, σ2, 
                                                         gp.decomposition)
    elseif transform_to_real && gp.decomposition == nothing
        err = """Need SVD decomposition to transform back to original space, 
                 but GPObj.decomposition == nothing. 
                 Try setting transform_to_real=false"""
        throw(ArgumentError(err))
    else
        μ_pred = μ
        # Convert to vector of (diagonal) matrices to match the format of 
        # σ2_pred when transform_to_real=true
        σ2_pred = vec([Diagonal(σ2[j, :]) for j in 1:N_new_inputs])
    end

    if output_dim == 1
        σ2_pred = reshape([σ2_pred[i][1] for i in 1:N_new_inputs], N_new_inputs, 1)
    end

    return μ_pred, σ2_pred
end


"""
svd_transform(data::Array{FT, 2}, obs_noise_cov::Union{Array{FT, 2}, Nothing}) where {FT}

Apply a singular value decomposition (SVD) to the data
  - `data` - GP training data/targets; N_samples x output_dim
  - `obs_noise_cov` - covariance of observational noise

Returns the transformed data and the decomposition, which is a matrix 
factorization of type LinearAlgebra.SVD. 
  
Note: If F::SVD is the factorization object, U, S, V and Vt can be obtained via 
F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt. The singular values 
in S are sorted in descending order.
"""

function svd_transform(data::Array{FT, 2}, obs_noise_cov::Union{Array{FT, 2}, Nothing}) where {FT}
    if obs_noise_cov != nothing
        decomposition = svd(obs_noise_cov)
        sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
        transformed_data_T = sqrt_singular_values_inv * decomposition.Vt * data'
        transformed_data = convert(Array, transformed_data_T')
    else
        decomposition = nothing
        transformed_data = data
    end

    return transformed_data, decomposition
end

function svd_transform(data::Vector{FT}, obs_noise_cov::Union{Array{FT, 2}, Nothing}) where {FT}
    if obs_noise_cov != nothing
        decomposition = svd(obs_noise_cov)
        sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
        transformed_data = sqrt_singular_values_inv * decomposition.Vt * data
    else
        decomposition = nothing
        transformed_data = data
    end

    return transformed_data, decomposition
end


"""
svd_reverse_transform_mean_cov(μ::Array{FT, 2}, σ2::{Array{FT, 2}, decomposition::SVD) where {FT}

Transform the mean and covariance back to the original (correlated) coordinate system
  - `μ` - predicted mean; N_predicted_points x output_dim
  - `σ2` - predicted variance; N_predicted_points x output_dim

Returns the transformed mean (N_predicted_points x output_dim) and variance. 
Note that transforming the variance back to the original coordinate system
results in non-zero off-diagonal elements, so instead of just returning the 
elements on the main diagonal (i.e., the variances), we return the full 
covariance at each point, as a vector of length N_predicted_points, where 
each element is a matrix of size output_dim x output_dim
"""

function svd_reverse_transform_mean_cov(μ::Array{FT, 2}, σ2::Array{FT, 2}, decomposition::SVD) where {FT}

    N_predicted_points, output_dim = size(σ2)
    # We created meanvGP = D_inv * Vt * mean_v so meanv = V * D * meanvGP
    sqrt_singular_values= Diagonal(sqrt.(decomposition.S)) 
    transformed_μT = decomposition.V * sqrt_singular_values * μ'
    # Get the means back into size N_prediction_points x output_dim
    transformed_μ = convert(Array, transformed_μT')
    transformed_σ2 = [zeros(output_dim, output_dim) for i in 1:N_predicted_points]
    # Back transformation
    for j in 1:N_predicted_points
        σ2_j = decomposition.V * sqrt_singular_values * Diagonal(σ2[j, :]) * sqrt_singular_values * decomposition.Vt
        transformed_σ2[j] = σ2_j
    end

    return transformed_μ, transformed_σ2
end

"""
  gp_sqbias(gp::GPObj{FT,GPJL}, inputs::Array{FT,2}, exp_outputs::Array{FT,2}) where {FT}
Evaluate the GP model(s) mean squared bias over a range of data points.
  - `gp` - a GPObj
  - `inputs` - inputs for which GP model(s) is/are evaluated; N_new_inputs x N_parameters
  - `exp_outputs` - outputs from the true model for the given inputs; N_new_inputs x N_outputs
"""
function gp_sqbias(gp::GPObj{FT, GPJL},
                  inputs::Array{FT,2},
                  exp_outputs::Array{FT,2}) where {FT}

    y_sqbias = zeros(length(exp_outputs[1,:]))
    N_data = length(inputs[:,1])
    for i in 1:N_data
        y_gp_mean, y_gp_var = predict(gp, reshape(inputs[i,:], 1, :) )
        y_gp_mean = dropdims(y_gp_mean, dims=1)
        y_sqbias = y_sqbias + (y_gp_mean-exp_outputs[i,:]).*(y_gp_mean-exp_outputs[i,:])
    end

    return y_sqbias./FT(N_data)
end

end 
