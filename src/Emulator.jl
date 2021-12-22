module Emulators

using ..DataStorage
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export Emulator
export Decomposition

export optimize_hyperparameters!
export predict

# SVD decomposition structure
struct Decomposition{FT<:AbstractFloat, IT<:Int}
    V::Array{FT,2}
    Vt::Array{FT,2}
    S::Array{FT}
    N::IT
end

# SVD decomposition constructor
function Decomposition(svd::SVD)
	# svd.V is of type adjoint, transformed to Array with [:,:]
	return Decomposition(svd.V[:,:], svd.Vt, svd.S, size(svd.S)[1])
end


abstract type MachineLearningTool end
# defaults in error, all MachineLearningTools require these functions.
function throw_define_mlt()
    throw(ErrorException("Unknown MachineLearningTool defined, please use a known implementation"))
end
function build_models!(mlt,iopairs) throw_define_mlt() end
function optimize_hyperparameters!(mlt) throw_define_mlt() end
function predict(mlt,new_inputs) throw_define_mlt() end

# include the different <: ML models
include("GaussianProcess.jl") #for GaussianProcess
# include("RandomFeature.jl")
# include("NeuralNetwork.jl")
# etc.


# We will define the different emulator types after the general statements

"""
    Emulator

Structure used to represent a general emulator
"""

struct Emulator{FT}
    "Machine learning tool, defined as a struct of type MachineLearningTool"
    machine_learning_tool::MachineLearningTool
     "normalized, standardized, transformed pairs given the Boolean's normalize_inputs, standardize_outputs, truncate_svd "
    training_pairs::PairedDataContainer{FT}
    "mean of input; 1 × input_dim"
    input_mean::Array{FT}
    "square root of the inverse of the input covariance matrix; input_dim × input_dim"
    normalize_inputs::Bool
    "whether to fit models on normalized outputs outputs / standardize_outputs_factor"
    sqrt_inv_input_cov::Union{Nothing, Array{FT, 2}}
    " if normalizing: whether to fit models on normalized inputs ((inputs - input_mean) * sqrt_inv_input_cov)"
    standardize_outputs::Bool
    "if standardizing: Standardization factors (characteristic values of the problem)"
    standardize_outputs_factors
    "the singular value decomposition of obs_noise_cov, such that obs_noise_cov = decomposition.U * Diagonal(decomposition.S) * decomposition.Vt. NB: the svd may be reduced in dimensions"
    decomposition::Union{Decomposition, Nothing}
end

# Constructor for the Emulator Object
function Emulator(
    machine_learning_tool::MachineLearningTool,
    input_output_pairs::PairedDataContainer{FT};
    obs_noise_cov=nothing,
    normalize_inputs::Bool = true,
    standardize_outputs::Bool = false,
    standardize_outputs_factors::Union{Array{FT,1}, Nothing}=nothing,
    truncate_svd::FT=1.0
    ) where {FT<:AbstractFloat}

    # For Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)
    if obs_noise_cov != nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end

    
    # [1.] Normalize the inputs? 
    input_mean = reshape(mean(get_inputs(input_output_pairs), dims=2), :, 1) #column vector
    sqrt_inv_input_cov = nothing
    if normalize_inputs
        # Normalize (NB the inputs have to be of) size [input_dim × N_samples] to pass to GPE())
        sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(input_output_pairs), dims=2))))
        training_inputs = normalize(
            get_inputs(input_output_pairs),
            input_mean,
            sqrt_inv_input_cov
        )
    else
        training_inputs = get_inputs(input_output_pairs)
    end

    # [2.] Standardize the outputs?
    if standardize_outputs
        training_outputs, obs_noise_cov = standardize(
            get_outputs(input_output_pairs),
            obs_noise_cov,
            standardize_outputs_factors
        )
    else
	training_outputs = get_outputs(input_output_pairs)
    end

    # [3.] Decorrelating the outputs [always performed]

    #Transform data if obs_noise_cov available 
    # (if obs_noise_cov==nothing, transformed_data is equal to data)
    decorrelated_training_outputs, decomposition = svd_transform(training_outputs, obs_noise_cov, truncate_svd=truncate_svd)

    # write new pairs structure 
    if truncate_svd<1.0
        #note this changes the dimension of the outputs
        training_pairs = PairedDataContainer(training_inputs, decorrelated_training_outputs)
        input_dim, output_dim = size(training_pairs, 1)
    else
        training_pairs = PairedDataContainer(training_inputs, decorrelated_training_outputs)
    end

    # [4.] build an emulator
    build_models!(machine_learning_tool,training_pairs)
    
    return Emulator{FT}(machine_learning_tool,
                        training_pairs,
                        input_mean,
                        normalize_inputs,
                        sqrt_inv_input_cov,
                        standardize_outputs,
                        standardize_outputs_factors,
                        decomposition)
end    


function optimize_hyperparameters!(emulator::Emulator{FT}) where {FT}
    optimize_hyperparameters!(emulator.machine_learning_tool)
end

function predict(emulator::Emulator{FT}, new_inputs; transform_to_real=false) where {FT}

    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(emulator.training_pairs, 1)

    N_samples = size(new_inputs, 2)
    size(new_inputs, 1) == input_dim || throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    # [1.] normalize
    normalized_new_inputs = normalize(emulator,new_inputs)

    # [2.]  predict. Note: ds = decorrelated, standard
    ds_outputs, ds_output_var = predict(emulator.machine_learning_tool, normalized_new_inputs)

    # [3.] transform back to real coordinates or remain in decorrelated coordinates
    if transform_to_real && emulator.decomposition == nothing
        throw(ArgumentError("""Need SVD decomposition to transform back to original space, 
                 but GaussianProcess.decomposition == nothing. 
                 Try setting transform_to_real=false"""))
    elseif transform_to_real && emulator.decomposition != nothing
        #transform back to real coords - cov becomes dense
        s_outputs, s_output_cov = svd_reverse_transform_mean_cov(
            ds_outputs, ds_output_var, emulator.decomposition)

        if output_dim == 1
            s_output_cov = [s_output_cov[i][1] for i in 1:N_samples]
        end
        # [4.] unstandardize
        return reverse_standardize(emulator, s_outputs, s_output_cov)
        
    else
        # remain in decorrelated, standardized coordinates (cov remains diagonal)
        # Convert to vector of  matrices to match the format  
        # when transform_to_real=true
        ds_output_diagvar = vec([Diagonal(ds_output_var[:, j]) for j in 1:N_samples])
        if output_dim == 1
            ds_output_diagvar = [ds_output_diagvar[i][1] for i in 1:N_samples]
        end

        return ds_outputs, ds_output_diagvar
        
    end

end




# Normalization, Standardization, and Decorrelation
function normalize(inputs, input_mean, sqrt_inv_input_cov)
    training_inputs = sqrt_inv_input_cov * (inputs .- input_mean)
    return training_inputs 
end

function normalize(emulator::Emulator{FT}, inputs) where {FT}
    if emulator.normalize_inputs
        return normalize(inputs, emulator.input_mean, emulator.sqrt_inv_input_cov)
    else
        return inputs
    end
end

function standardize(outputs, output_cov, factors) 
    standardized_outputs = outputs ./ factors
    standardized_cov = output_cov ./ (factors .* factors')
    return standardized_outputs, standardized_cov
end

function reverse_standardize(emulator::Emulator{FT}, outputs, output_cov) where {FT}
    if emulator.standardize_outputs
        return standardize(outputs, output_cov, 1. / emulator.standardize_output_factors)
    else
        return outputs, output_cov
    end
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
function svd_transform(
    data::Array{FT, 2},
    obs_noise_cov; 
    truncate_svd::FT=1.0) where {FT}

    if obs_noise_cov != nothing
        @assert ndims(obs_noise_cov) == 2
    end
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
            decomposition = Decomposition(decomposition.V[:,1:k], decomposition.Vt[1:k,:], 
                                   decomposition.S[1:k], n)
	else
            decomposition = svd(obs_noise_cov)
            sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
            transformed_data = sqrt_singular_values_inv * decomposition.Vt * data
	    decomposition = Decomposition(svd(obs_noise_cov))
        end
    else
        decomposition = nothing
        transformed_data = data
    end

    return transformed_data, decomposition
end


"""
function svd_transform(
    data::Vector{FT}, 
    obs_noise_cov::Union{Array{FT, 2}, Nothing};
    truncate_svd::FT=1.0) where {FT}

"""
function svd_transform(
    data::Vector{FT}, 
    obs_noise_cov;
    truncate_svd::FT=1.0) where {FT}

   
    if obs_noise_cov != nothing
        @assert ndims(obs_noise_cov) == 2
    end
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
            decomposition = Decomposition(decomposition.V[:,1:k], decomposition.Vt[1:k,:], 
                                   decomposition.S[1:k], n)
	else
            decomposition = svd(obs_noise_cov)
            sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomposition.S)) 
            transformed_data = sqrt_singular_values_inv * decomposition.Vt * data
	    decomposition = Decomposition(svd(obs_noise_cov))
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
function svd_reverse_transform_mean_cov(μ, σ2, 
                                        decomposition::Union{SVD, Decomposition};
					truncate_svd::FT=1.0) where {FT}
    @assert ndims(μ) == 2
    @assert ndims(σ2) == 2
    
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
