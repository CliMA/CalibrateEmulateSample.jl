
using GaussianProcesses
   
using PyCall
using ScikitLearn
const pykernels = PyNULL()
const pyGP = PyNULL()
function __init__()
    copy!(pykernels, pyimport("sklearn.gaussian_process.kernels"))
    copy!(pyGP, pyimport("sklearn.gaussian_process"))
end

#exports (from Emulator)
export GaussianProcess

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
    GaussianProcess{FT<:AbstractFloat} <: MachineLearningTool

    Structure holding training input and the fitted Gaussian Process Regression
models.

# Fields
$(DocStringExtensions.FIELDS)

"""
struct GaussianProcess{GPPackage} <: MachineLearningTool
    "the Gaussian Process (GP) Regression model(s) that are fitted to the given input-data pairs"
    models::Vector
    "Kernel object"
    kernel
    "learn the noise with the White Noise kernel explicitly?"
    noise_learn
    "prediction type (`y` to predict the data, `f` to predict the latent function)"
    prediction_type::Union{Nothing, PredictionType}
end


"""
    GaussianProcess(input_output_pairs::PairedDataContainer{FT}, package::GPJL; kernel::Union{K, KPy, Nothing}=nothing, obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing, normalized::Bool=true, noise_learn::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Input-output pairs in paired data storage, storing in/out_dim × N_samples

 - `kernel` - GaussianProcesses kernel object. If not supplied, a default Squared Exponential kernel is used.
"""

function GaussianProcess(
    package;
    kernel::Union{K, KPy, Nothing}=nothing,
    noise_learn=nothing,
    prediction_type::PredictionType=YType(),
    ) where {K<:Kernel, KPy<:PyObject}

    # Initialize vector for GP models
    models = Any[]
    
    return GaussianProcess{typeof(package)}(models, kernel, noise_learn, prediction_type)
			
end

# First we create  the GPJL implementation
function build_models!(
    gp::GaussianProcess{GPJL},
    input_output_pairs)
    
    # get inputs and outputs 
    input_values = get_inputs(input_output_pairs)
    output_values = get_outputs(input_output_pairs)

    # Number of models (We are fitting one model per output dimension, as data is decorrelated)
    models = gp.models
    N_models = size(output_values,1); #size(transformed_data)[1]


    # Use a default kernel unless a kernel was supplied to GaussianProcess
    if gp.kernel==nothing
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
        kernel_i = deepcopy(kern)
        println(kernel_i)
        data_i = output_values[i,:]
        # GPE() arguments:
        # input_values:    (input_dim × N_samples)
        # GPdata_i:    (N_samples,)
       
        # Zero mean function
        kmean = MeanZero()

        # Instantiate GP model
        m = GPE(input_values,
                output_values[i,:],
                kmean,
                kernel_i, 
                logstd_regularization_noise)
       
        println("created GP", i)
        push!(models, m)
        
    end

end

function optimize_hyperparameters!(gp::GaussianProcess{GPJL})
    N_models = length(gp.models)
    for i = 1:N_models
        #noise_true means we do it explicitly (thus don't need it here)
        optimize!(gp.models[i], noise=!gp.noise_learn)
        println("optimized hyperparameters of GP", i)
        println(gp.models[i].kernel)
    end
end

predict(gp::GaussianProcess{GPJL}, new_inputs::Array{FT, 2}) where {FT} = predict(gp, new_inputs, gp.prediction_type)

function predict(gp::GaussianProcess{GPJL}, new_inputs::Array{FT, 2}, ::YType) where {FT}


    M = length(gp.models)
    N_samples = size(new_inputs,2)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(M, N_samples)
    σ2 = zeros(M, N_samples)
    for i in 1:M
        μ[i,:],σ2[i,:] = predict_y(gp.models[i], new_inputs)
    end

    return μ, σ2
end



function predict(gp::GaussianProcess{GPJL}, new_inputs::Array{FT, 2}, ::FType) where {FT}

    M = length(gp.models)
    N_samples = size(new_inputs,2)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(M, N_samples)
    σ2 = zeros(M, N_samples)
    for i in 1:M
        μ[i,:],σ2[i,:] = predict_f(gp.models[i], new_inputs)
    end

    return μ, σ2
end




"""
    GaussianProcess(input_output_pairs::PairedDataContainer, package::SKLJL; kernel::Union{K, KPy, Nothing}=nothing, obs_noise_cov::Union{Array{FT, 2}, Nothing}=nothing, normalized::Bool=true, noise_learn::Bool=true, prediction_type::PredictionType=YType()) where {K<:Kernel, KPy<:PyObject}

Input-output pairs in paired data storage, storing in/out_dim × N_samples

 - `kernel` - ScikitLearn kernel object. If not supplied, a default Squared Exponential kernel is used.
"""
#now we build the SKLJL implementation

function build_models!(
    gp::GaussianProcess{SKLJL},
    input_output_pairs;
    noise_learn::Bool=true)

      # get inputs and outputs 
    input_values = permutedims(get_inputs(input_output_pairs), (2,1))
    output_values = get_outputs(input_output_pairs)

    # Number of models (We are fitting one model per output dimension, as data is decorrelated)
    models = gp.models
    N_models = size(output_values,1); #size(transformed_data)[1]

    if gp.kernel==nothing
        println("Using default squared exponential kernel, learning length scale and variance parameters")
        # Create default squared exponential kernel
        const_value = 1.0
        var_kern = pykernels.ConstantKernel(constant_value=const_value,
                                            constant_value_bounds=(1e-5, 1e4))
        rbf_len = ones(size(input_values, 2))
        rbf = pykernels.RBF(length_scale=rbf_len, 
                            length_scale_bounds=(1e-5, 1e4))
        kern = var_kern * rbf
        println("Using default squared exponential kernel:", kern)
    else
        kern = deepcopy(gp.kernel)
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
        kernel_i = deepcopy(kern)
        data_i = transformed_data[i, :]
        m = pyGP.GaussianProcessRegressor(kernel=kernel_i,
                                          n_restarts_optimizer=10,
                                          alpha=regularization_noise)

        # ScikitLearn.fit! arguments:
        # input_values:    (N_samples × input_dim)
        # data_i:    (N_samples,)
        ScikitLearn.fit!(m, input_values, data_i)
        if i==1
            println(m.kernel.hyperparameters)
            print("Completed training of: ")
        end
        println(i, ", ")
        push!(models, m)
        println(m.kernel)
    end
end


function optimize_hyperparameters!(gp::GaussianProcess{SKLJL})
    println("SKlearn, already trained. continuing...")
end


function predict(gp::GaussianProcess{SKLJL}, new_inputs::Array{FT, 2}) where {FT}

    M = length(gp.models)
    N_samples = size(new_inputs,2)
    
    # SKJL based on rows not columns; need to transpose inputs
    μ = zeros(M, N_samples)
    σ = zeros(M, N_samples)
    for i in 1:M
        μ[i,:],σ[i,:] = gp.models[i].predict(new_inputs', return_std=true)
    end
    σ2 = σ .* σ

    return μ, σ2
end

#=
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
=#
