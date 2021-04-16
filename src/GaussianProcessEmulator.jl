module GaussianProcessEmulator

using Statistics
using Distributions
using LinearAlgebra
using GaussianProcesses
using DocStringExtensions
using ..DataStorage
    
using PyCall
using ScikitLearn
const pykernels = PyNULL()
const pyGP = PyNULL()
function __init__()
    copy!(pykernels, pyimport("sklearn.gaussian_process.kernels"))
    copy!(pyGP, pyimport("sklearn.gaussian_process"))
end

export GaussianProcess
export predict

export GPJL, SKLJL
export YType, FType
export svd_transform, svd_reverse_transform_mean_cov
export decomp_struct

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


# SVD decomposition structure
struct decomp_struct{FT<:AbstractFloat, IT<:Int}
    V::Array{FT,2}
    Vt::Array{FT,2}
    S::Array{FT}
    N::IT
end
# SVD decomposition constructor
function decomp_struct(svd::SVD)
	# svd.V is of type adjoint, transformed to Array with [:,:]
	return decomp_struct(svd.V[:,:], svd.Vt, svd.S, size(svd.S)[1])
end

"""
    GaussianProcess{FT<:AbstractFloat}

    Structure holding training input and the fitted Gaussian Process Regression
models.

# Fields
$(DocStringExtensions.FIELDS)

"""
struct GaussianProcess{FT<:AbstractFloat, GPM}
    "training inputs and outputs, data stored in columns"
    input_output_pairs::PairedDataContainer{FT}
    "mean of input; 1 × input_dim"
    input_mean::Array{FT}
    "square root of the inverse of the input covariance matrix; input_dim × input_dim"
    sqrt_inv_input_cov::Union{Nothing, Array{FT, 2}}
    "the Gaussian Process (GP) Regression model(s) that are fitted to the given input-data pairs"
    models::Vector
    "the singular value decomposition of obs_noise_cov, such that obs_noise_cov = decomposition.U * Diagonal(decomposition.S) * decomposition.Vt."
    decomposition::Union{decomp_struct, Nothing}
    "whether to fit GP models on normalized inputs ((inputs - input_mean) * sqrt_inv_input_cov)"
    normalized::Bool
    "prediction type (`y` to predict the data, `f` to predict the latent function)"
    prediction_type::Union{Nothing, PredictionType}
    "Standardization factors (characteristic values of the problem)"
    norm_factors::Union{Array{FT}, Nothing}
end


"""
    GaussianProcess(input_output_pairs::PairedDataContainer{FT}, package::GPJL; GPkernel::Union{K, KPy, Nothing}=nothing, obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing, normalized::Bool=true, noise_learn::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Input-output pairs in paired data storage, storing in/out_dim × N_samples

 - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default Squared Exponential kernel is used.
"""

function GaussianProcess(
    input_output_pairs::PairedDataContainer{FT},
    package::GPJL;
    GPkernel::Union{K, KPy, Nothing}=nothing,
    obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing,
    normalized::Bool=true,
    noise_learn::Bool=true,
    truncate_svd::FT=1.0,
    standardize::Bool=false,
    prediction_type::PredictionType=YType(),
    norm_factor::Union{Array{FT,1}, Nothing}=nothing) where {FT<:AbstractFloat, K<:Kernel, KPy<:PyObject}

    # Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)
    
    if obs_noise_cov != nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end

    # Initialize vector of GP models
    models = Any[]
   

    # Can use the full time median or some user define function
    if standardize
	output_values = get_outputs(input_output_pairs) ./ norm_factor
	obs_noise_cov = obs_noise_cov ./ (norm_factor .* norm_factor')
    else
	output_values = get_outputs(input_output_pairs)
    end

    # Normalize the inputs if normalized==true 
    input_mean = reshape(mean(get_inputs(input_output_pairs), dims=2), :, 1) #column vector
    sqrt_inv_input_cov = nothing
    if normalized
        # Normalize (NB the inputs have to be of) size [input_dim × N_samples] to pass to GPE())
        sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(input_output_pairs), dims=2))))
        GPinputs = sqrt_inv_input_cov * (get_inputs(input_output_pairs).-input_mean)
    else
        GPinputs = get_inputs(input_output_pairs)
    end
    
    
    # Transform data if obs_noise_cov available 
    # (if obs_noise_cov==nothing, transformed_data is equal to data)
    transformed_data, decomposition = svd_transform(output_values, 
						    obs_noise_cov, truncate_svd=truncate_svd)

    # Overwrite the input-output pairs because of the size discrepency
    if truncate_svd<1.0
	# Overwrite the input_output_pairs structure
	input_output_pairs_old = deepcopy(input_output_pairs)
	input_output_pairs = PairedDataContainer(get_inputs(input_output_pairs_old),
					      transformed_data, data_are_columns=true)
        input_dim, output_dim = size(input_output_pairs, 1)
    end

    # Number of models (We are fitting one model per output dimension)
    N_models = output_dim; #size(transformed_data)[1]
    
    # Use a default kernel unless a kernel was supplied to GaussianProcess
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
        println("kernel in GaussianProcess")
        GPkernel_i = deepcopy(kern)
        println(GPkernel_i)
        GPdata_i = transformed_data[i,:]
        # GPE() arguments:
        # GPinputs:    (input_dim × N_samples)
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
    
    return GaussianProcess{FT,typeof(package)}(input_output_pairs,
                                               input_mean, 
                                               sqrt_inv_input_cov,
                                               models,
                                               decomposition,
                                               normalized,
                                               prediction_type,
					       norm_factor)
end


"""
    GaussianProcess(input_output_pairs::PairedDataContainer, package::SKLJL; GPkernel::Union{K, KPy, Nothing}=nothing, obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing, normalized::Bool=true, noise_learn::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Input-output pairs in paired data storage, storing in/out_dim × N_samples

 - `GPkernel` - ScikitLearn kernel object. If not supplied, a default Squared Exponential kernel is used.
"""
function GaussianProcess(
    input_output_pairs::PairedDataContainer{FT},
    package::SKLJL;
    GPkernel::Union{K, KPy, Nothing}=nothing,
    obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing,
    normalized::Bool=true,
    noise_learn::Bool=true,
    truncate_svd::FT=1.0,
    standardize::Bool=false,
    prediction_type::PredictionType=YType(),
    norm_factor::Union{Array{FT,1}, Nothing}=nothing) where {FT<:AbstractFloat, K<:Kernel, KPy<:PyObject}

    # Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)

    if obs_noise_cov != nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end

    # Initialize vector of GP models
    models = Any[]
    # Number of models (We are fitting one model per output dimension)
    N_models = output_dim

    # Can use the full time median or some user define function
    if standardize
        #norm_factors = get_standarizing_factors() 
	output_values = get_outputs(input_output_pairs) ./ norm_factor
	obs_noise_cov = obs_noise_cov ./ (norm_factor .* norm_factor')
    else
	output_values = get_outputs(input_output_pairs)
    end

    
    # Normalize the inputs if normalized==true
    # Note that contrary to the GaussianProcesses.jl (GPJL) GPE, the 
    # ScikitLearn (SKLJL) GaussianProcessRegressor requires inputs to be of 
    # size N_samples × input_dim, so one needs to transpose
    input_mean = reshape(mean(get_inputs(input_output_pairs), dims=2), :, 1)
    sqrt_inv_input_cov = nothing
    if normalized
        sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(input_output_pairs), dims=2))))
        GPinputs = permutedims(sqrt_inv_input_cov*(get_inputs(input_output_pairs) .- input_mean), (2,1))
    else
        GPinputs = permutedims(get_inputs(input_output_pairs), (2,1))
    end

    # Transform data if obs_noise_cov available (if obs_noise_cov==nothing, 
    # transformed_data is equal to data)
    transformed_data, decomposition = svd_transform(output_values, 
						    obs_noise_cov, truncate_svd=truncate_svd)

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
        GPdata_i = transformed_data[i, :]
        m = pyGP.GaussianProcessRegressor(kernel=GPkernel_i,
                                          n_restarts_optimizer=10,
                                          alpha=regularization_noise)

        # ScikitLearn.fit! arguments:
        # GPinputs:    (N_samples × input_dim)
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
    return GaussianProcess{FT, typeof(package)}(input_output_pairs,
                                                input_mean,
                                                sqrt_inv_input_cov,
                                                models,
                                                decomposition,
                                                normalized,
                                                YType(),
						norm_factor)
end


"""
    predict(gp::GaussianProcess{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real::Bool=false) where {FT} = predict(gp, new_inputs, gp.prediction_type)

Evaluate the GP model(s) at new inputs.
  - `gp` - a GaussianProcess
  - `new_inputs` - inputs for which GP model(s) is/are evaluated; input_dim × N_samples

Returns the predicted mean(s) and covariance(s) at the input points. 
Means: matrix of size output_dim × N_samples 
Covariances: vector of length N_samples, each element is a matrix of size output_dim × output_dim. If the output is 1-dimensional, a 1 × N_samples array of scalar variances is returned rather than a vector of 1x1 matrices.

Note: If gp.normalized == true, the new inputs are normalized prior to the prediction
"""
predict(gp::GaussianProcess{FT, GPJL}, new_inputs::Array{FT, 2}; transform_to_real::Bool=false) where {FT} = predict(gp, new_inputs, transform_to_real, gp.prediction_type)

function predict(gp::GaussianProcess{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real, ::FType) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(gp.input_output_pairs, 1)
    N_samples = size(new_inputs, 2)
    size(new_inputs, 1) == input_dim || throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    if gp.normalized
        new_inputs = gp.sqrt_inv_input_cov * (new_inputs .- gp.input_mean)  
    end

    M = length(gp.models)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(output_dim,N_samples)
    σ2 = zeros(output_dim,N_samples)
    for i in 1:M
        μ[i,:],σ2[i,:] = predict_f(gp.models[i], new_inputs)
    end

    if transform_to_real && gp.decomposition != nothing
        μ_pred, σ2_pred = svd_reverse_transform_mean_cov(μ, σ2,
                                                         gp.decomposition)
    elseif transform_to_real && gp.decomposition == nothing
        throw(ArgumentError("""Need SVD decomposition to transform back to original space, 
                 but GaussianProcess.decomposition == nothing. 
                 Try setting transform_to_real=false"""))
    else
        μ_pred = μ
        # Convert to vector of (diagonal) matrices to match the format of 
        # σ2_pred when transform_to_real=true
        σ2_pred = vec([Diagonal(σ2[:, j]) for j in 1:N_samples])
    end

    if output_dim == 1
        σ2_pred = [σ2_pred[i][1] for i in 1:N_samples]
    end

    return μ_pred, σ2_pred
end

function predict(gp::GaussianProcess{FT, GPJL}, new_inputs::Array{FT, 2}, transform_to_real, ::YType) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(gp.input_output_pairs, 1)

    N_samples = size(new_inputs, 2)
    size(new_inputs, 1) == input_dim || throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    if gp.normalized
        new_inputs = gp.sqrt_inv_input_cov * (new_inputs .- gp.input_mean)  
    end

    M = length(gp.models)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(output_dim,N_samples)
    σ2 = zeros(output_dim,N_samples)
    for i in 1:M
        μ[i,:],σ2[i,:] = predict_f(gp.models[i], new_inputs)
    end
    if transform_to_real && gp.decomposition != nothing
        μ_pred, σ2_pred = svd_reverse_transform_mean_cov(μ, σ2, 
                                                         gp.decomposition)
    elseif transform_to_real && gp.decomposition == nothing
        throw(ArgumentError("""Need SVD decomposition to transform back to original space, 
                 but GaussianProcess.decomposition == nothing. 
                 Try setting transform_to_real=false"""))
    else
        μ_pred = μ
        # Convert to vector of (diagonal) matrices to match the format of 
        # σ2_pred when transform_to_real=true
        σ2_pred = vec([Diagonal(σ2[:, j]) for j in 1:N_samples])
    end

    if output_dim == 1
        σ2_pred = [σ2_pred[i][1] for i in 1:N_samples]
    end

    return μ_pred, σ2_pred
end

function predict(gp::GaussianProcess{FT, SKLJL}, new_inputs::Array{FT, 2}, transform_to_real::Bool=false) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(gp.input_output_pairs, 1)

    N_samples = size(new_inputs, 2)
    size(new_inputs, 1) == input_dim || throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    if gp.normalized
        new_inputs = gp.sqrt_inv_input_cov * (new_inputs .- gp.input_mean)  
    end

    M = length(gp.models)

    # SKJL based on rows not columns; need to transpose inputs
    μ = zeros(output_dim,N_samples)
    σ = zeros(output_dim,N_samples)
    for i in 1:M
        μ[i,:],σ[i,:] = gp.models[i].predict(new_inputs', return_std=true)
    end
    σ2 = σ .* σ

    if transform_to_real && gp.decomposition != nothing
        μ_pred, σ2_pred = svd_reverse_transform_mean_cov(μ, σ2, 
                                                         gp.decomposition)
    elseif transform_to_real && gp.decomposition == nothing
        throw(ArgumentError("""Need SVD decomposition to transform back to original space, 
                 but GaussianProcess.decomposition == nothing. 
                 Try setting transform_to_real=false"""))
    else
        μ_pred = μ
        # Convert to vector of (diagonal) matrices to match the format of 
        # σ2_pred when transform_to_real=true
        σ2_pred = vec([Diagonal(σ2[:, j]) for j in 1:N_samples])
    end

    if output_dim == 1
        σ2_pred = reshape([σ2_pred[i][1] for i in 1:N_samples], 1,N_samples)
    end

    return μ_pred, σ2_pred
end

"""
svd_transform(data::Array{FT, 2}, obs_noise_cov::Union{Array{FT, 2}, Nothing}) where {FT}

Apply a singular value decomposition (SVD) to the data
  - `data` - GP training data/targets; output_dim × N_samples
  - `obs_noise_cov` - covariance of observational noise

Returns the transformed data and the decomposition, which is a matrix 
factorization of type LinearAlgebra.SVD. 
  
Note: If F::SVD is the factorization object, U, S, V and Vt can be obtained via 
F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt. The singular values 
in S are sorted in descending order.
"""
function svd_transform(data::Array{FT, 2}, obs_noise_cov::Union{Array{FT, 2}, Nothing}; 
		       truncate_svd::FT=1.0) where {FT}
    if obs_noise_cov != nothing
        # Truncate the SVD as a form of regularization
	if truncate_svd<1.0 # this variable needs to be provided to this function
            # Perform SVD
            decomposition = svd(obs_noise_cov)
            # Find cutoff
            σ = decomposition.S
            σ_cumsum = cumsum(σ) / sum(σ);
            P_cutoff = truncate_svd;
            ind = findall(x->x>P_cutoff, σ_cumsum); k = ind[1]
            println("SVD truncated at k:")
            println(k)
            # Apply truncated SVD
            n = size(obs_noise_cov)[1]
            sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S))
	    transformed_data = sqrt_singular_values_inv[1:k,1:k] * decomposition.Vt[1:k,:] * data
            transformed_data = transformed_data;
            decomposition = decomp_struct(decomposition.V[:,1:k], decomposition.Vt[1:k,:], 
                                   decomposition.S[1:k], n)
	else
            decomposition = svd(obs_noise_cov)
            sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
            transformed_data = sqrt_singular_values_inv * decomposition.Vt * data
	    decomposition = decomp_struct(svd(obs_noise_cov))
        end
    else
        decomposition = nothing
        transformed_data = data
    end

    return transformed_data, decomposition
end

function svd_transform(data::Vector{FT}, 
		       obs_noise_cov::Union{Array{FT, 2}, Nothing};
		       truncate_svd::FT=1.0) where {FT}
     if obs_noise_cov != nothing
        # Truncate the SVD as a form of regularization
	if truncate_svd<1.0 # this variable needs to be provided to this function
            # Perform SVD
            decomposition = svd(obs_noise_cov)
            # Find cutoff
            σ = decomposition.S
            σ_cumsum = cumsum(σ) / sum(σ);
            P_cutoff = truncate_svd;
            ind = findall(x->x>P_cutoff, σ_cumsum); k = ind[1]
            println("SVD truncated at k:")
            println(k)
            # Apply truncated SVD
            n = size(obs_noise_cov)[1]
            sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S))
	    transformed_data = sqrt_singular_values_inv[1:k,1:k] * decomposition.Vt[1:k,:] * data
            transformed_data = transformed_data;
            decomposition = decomp_struct(decomposition.V[:,1:k], decomposition.Vt[1:k,:], 
                                   decomposition.S[1:k], n)
	else
            decomposition = svd(obs_noise_cov)
            sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
            transformed_data = sqrt_singular_values_inv * decomposition.Vt * data
	    decomposition = decomp_struct(svd(obs_noise_cov))
        end
    else
        decomposition = nothing
        transformed_data = data
    end

    return transformed_data, decomposition
end

"""
svd_reverse_transform_mean_cov(μ::Array{FT, 2}, σ2::{Array{FT, 2}, decomposition::SVD) where {FT}

Transform the mean and covariance back to the original (correlated) coordinate system
  - `μ` - predicted mean; output_dim × N_predicted_points
  - `σ2` - predicted variance; output_dim × N_predicted_points 

Returns the transformed mean (output_dim × N_predicted_points) and variance. 
Note that transforming the variance back to the original coordinate system
results in non-zero off-diagonal elements, so instead of just returning the 
elements on the main diagonal (i.e., the variances), we return the full 
covariance at each point, as a vector of length N_predicted_points, where 
each element is a matrix of size output_dim × output_dim
"""
function svd_reverse_transform_mean_cov(μ::Array{FT, 2}, σ2::Array{FT, 2}, 
                                        decomposition::Union{SVD, decomp_struct};
					truncate_svd::FT=1.0) where {FT}
        output_dim, N_predicted_points = size(σ2)
        # We created meanvGP = D_inv * Vt * mean_v so meanv = V * D * meanvGP
        sqrt_singular_values= Diagonal(sqrt.(decomposition.S))
        transformed_μ = decomposition.V * sqrt_singular_values * μ

        transformed_σ2 = [zeros(output_dim, output_dim) for i in 1:N_predicted_points]
        # Back transformation

        for j in 1:N_predicted_points
            σ2_j = decomposition.V * sqrt_singular_values * Diagonal(σ2[:,j]) * sqrt_singular_values * decomposition.Vt
            transformed_σ2[j] = σ2_j
        end

    return transformed_μ, transformed_σ2
end


end 
