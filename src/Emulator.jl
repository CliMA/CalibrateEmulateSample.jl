module Emulators

using ..DataContainers

using DocStringExtensions
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export Emulator

export optimize_hyperparameters!
export predict

"""
$(DocStringExtensions.TYPEDEF)

Type to dispatch different emulators:

 - GaussianProcess <: MachineLearningTool
"""
abstract type MachineLearningTool end
# defaults in error, all MachineLearningTools require these functions.
function throw_define_mlt()
    throw(ErrorException("Unknown MachineLearningTool defined, please use a known implementation"))
end
function build_models!(mlt, iopairs)
    throw_define_mlt()
end
function optimize_hyperparameters!(mlt)
    throw_define_mlt()
end
function predict(mlt, new_inputs)
    throw_define_mlt()
end

# include the different <: ML models
include("GaussianProcess.jl") #for GaussianProcess
# include("RandomFeature.jl")
# include("NeuralNetwork.jl")
# etc.


# We will define the different emulator types after the general statements

"""
$(DocStringExtensions.TYPEDEF)

Structure used to represent a general emulator, independently of the algorithm used.

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
struct Emulator{FT <: AbstractFloat}
    "Machine learning tool, defined as a struct of type MachineLearningTool."
    machine_learning_tool::MachineLearningTool
    "Normalized, standardized, transformed pairs given the Booleans normalize\\_inputs, standardize\\_outputs, retained\\_svd\\_frac."
    training_pairs::PairedDataContainer{FT}
    "Mean of input; length *input\\_dim*."
    input_mean::AbstractVector{FT}
    "Square root of the inverse of the input covariance matrix; size *input\\_dim* × *input\\_dim*."
    normalize_inputs::Bool
    "Whether to fit models on normalized outputs: `outputs / standardize_outputs_factor`."
    sqrt_inv_input_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing}
    "If normalizing: whether to fit models on normalized inputs (`(inputs - input_mean) * sqrt_inv_input_cov`)."
    standardize_outputs::Bool
    "If standardizing: Standardization factors (characteristic values of the problem)."
    standardize_outputs_factors::Union{AbstractVector{FT}, Nothing}
    "The singular value decomposition of *obs\\_noise\\_cov*, such that *obs\\_noise\\_cov* = decomposition.U * Diagonal(decomposition.S) * decomposition.Vt. NB: the SVD may be reduced in dimensions."
    decomposition::Union{SVD, Nothing}
    "Fraction of singular values kept in decomposition. A value of 1 implies full SVD spectrum information."
    retained_svd_frac::FT
end

# Constructor for the Emulator Object
function Emulator(
    machine_learning_tool::MachineLearningTool,
    input_output_pairs::PairedDataContainer{FT};
    obs_noise_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}, Nothing} = nothing,
    normalize_inputs::Bool = true,
    standardize_outputs::Bool = false,
    standardize_outputs_factors::Union{AbstractVector{FT}, Nothing} = nothing,
    retained_svd_frac::FT = 1.0,
) where {FT <: AbstractFloat}

    # For Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)
    if isa(obs_noise_cov, UniformScaling)
        # Cast UniformScaling to Diagonal, since UniformScaling incompatible with 
        # SVD and standardize()
        obs_noise_cov = Diagonal(obs_noise_cov, output_dim)
    end
    if obs_noise_cov !== nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end


    # [1.] Normalize the inputs? 
    input_mean = vec(mean(get_inputs(input_output_pairs), dims = 2)) #column vector
    sqrt_inv_input_cov = nothing
    if normalize_inputs
        # Normalize (NB the inputs have to be of) size [input_dim × N_samples] to pass to ML tool
        sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(input_output_pairs), dims = 2))))
        training_inputs = normalize(get_inputs(input_output_pairs), input_mean, sqrt_inv_input_cov)
    else
        training_inputs = get_inputs(input_output_pairs)
    end

    # [2.] Standardize the outputs?
    if standardize_outputs
        training_outputs, obs_noise_cov =
            standardize(get_outputs(input_output_pairs), obs_noise_cov, standardize_outputs_factors)
    else
        training_outputs = get_outputs(input_output_pairs)
    end

    # [3.] Decorrelating the outputs [always performed]

    #Transform data if obs_noise_cov available 
    # (if obs_noise_cov==nothing, transformed_data is equal to data)
    decorrelated_training_outputs, decomposition =
        svd_transform(training_outputs, obs_noise_cov, retained_svd_frac = retained_svd_frac)

    # write new pairs structure 
    if retained_svd_frac < 1.0
        #note this changes the dimension of the outputs
        training_pairs = PairedDataContainer(training_inputs, decorrelated_training_outputs)
        input_dim, output_dim = size(training_pairs, 1)
    else
        training_pairs = PairedDataContainer(training_inputs, decorrelated_training_outputs)
    end

    # [4.] build an emulator
    build_models!(machine_learning_tool, training_pairs)

    return Emulator{FT}(
        machine_learning_tool,
        training_pairs,
        input_mean,
        normalize_inputs,
        sqrt_inv_input_cov,
        standardize_outputs,
        standardize_outputs_factors,
        decomposition,
        retained_svd_frac,
    )
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimizes the hyperparameters in the machine learning tool.
"""
function optimize_hyperparameters!(emulator::Emulator{FT}) where {FT <: AbstractFloat}
    optimize_hyperparameters!(emulator.machine_learning_tool)
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes a prediction using the emulator on new inputs (each new inputs given as data columns).
Default is to predict in the decorrelated space.
"""
function predict(
    emulator::Emulator{FT},
    new_inputs::AbstractMatrix{FT};
    transform_to_real = false,
) where {FT <: AbstractFloat}
    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(emulator.training_pairs, 1)

    N_samples = size(new_inputs, 2)
    size(new_inputs, 1) == input_dim ||
    throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    # [1.] normalize
    normalized_new_inputs = normalize(emulator, new_inputs)

    # [2.]  predict. Note: ds = decorrelated, standard
    ds_outputs, ds_output_var = predict(emulator.machine_learning_tool, normalized_new_inputs)

    # [3.] transform back to real coordinates or remain in decorrelated coordinates
    if transform_to_real && emulator.decomposition === nothing
        throw(ArgumentError("""Need SVD decomposition to transform back to original space, 
                 but GaussianProcess.decomposition == nothing. 
                 Try setting transform_to_real=false"""))
    elseif transform_to_real && emulator.decomposition !== nothing
        #transform back to real coords - cov becomes dense
        s_outputs, s_output_cov = svd_reverse_transform_mean_cov(ds_outputs, ds_output_var, emulator.decomposition)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Normalize the input data, with a normalizing function.
"""
function normalize(emulator::Emulator{FT}, inputs::AbstractVecOrMat{FT}) where {FT <: AbstractFloat}
    if emulator.normalize_inputs
        return normalize(inputs, emulator.input_mean, emulator.sqrt_inv_input_cov)
    else
        return inputs
    end
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Normalize with the empirical Gaussian distribution of points.
"""
function normalize(
    inputs::AbstractVecOrMat{FT},
    input_mean::AbstractVector{FT},
    sqrt_inv_input_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
) where {FT <: AbstractFloat}
    training_inputs = sqrt_inv_input_cov * (inputs .- input_mean)
    return training_inputs
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Standardize with a vector of factors (size equal to output dimension).
"""
function standardize(
    outputs::AbstractVecOrMat{FT},
    output_covs::Vector{<:Union{AbstractMatrix{FT}, UniformScaling{FT}}},
    factors::AbstractVector{FT},
) where {FT <: AbstractFloat}
    # Case where `output_covs` is a Vector of covariance matrices
    n = length(factors)
    outputs = outputs ./ factors
    cov_factors = factors .* factors'
    for i in 1:length(output_covs)
        if isa(output_covs[i], Matrix)
            output_covs[i] = output_covs[i] ./ cov_factors
        elseif isa(output_covs[i], UniformScaling)
            # assert all elements of factors are same, otherwise can't cast back to
            # UniformScaling
            @assert all(y -> y == first(factors), factors)
            output_covs[i] = output_covs[i] / cov_factors[1, 1]
        else
            # Diagonal, TriDiagonal etc. case
            # need to do ./ as dense matrix and cast back to original type
            # https://discourse.julialang.org/t/get-generic-constructor-of-parametric-type/57189/8
            T = typeof(output_covs[i])
            cast_T = getfield(parentmodule(T), nameof(T))
            output_covs[i] = cast_T(Matrix(output_covs[i]) ./ cov_factors)
        end
    end
    return outputs, output_covs
end

function standardize(
    outputs::AbstractVecOrMat{FT},
    output_cov::Union{AbstractMatrix{FT}, UniformScaling{FT}},
    factors::AbstractVector{FT},
) where {FT <: AbstractFloat}
    # Case where `output_cov` is a single covariance matrix
    stdized_out, stdized_covs = standardize(outputs, [output_cov], factors)
    return stdized_out, stdized_covs[1]
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reverse a previous standardization with the stored vector of factors (size equal to output 
dimension). `output_cov` is a Vector of covariance matrices, such as is returned by
[`svd_reverse_transform_mean_cov`](@ref).
"""
function reverse_standardize(
    emulator::Emulator{FT},
    outputs::AbstractVecOrMat{FT},
    output_covs::Union{AbstractMatrix{FT}, Vector{<:AbstractMatrix{FT}}},
) where {FT <: AbstractFloat}
    if emulator.standardize_outputs
        return standardize(outputs, output_covs, 1.0 ./ emulator.standardize_outputs_factors)
    else
        return outputs, output_covs
    end
end



"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply a singular value decomposition (SVD) to the data
  - `data` - GP training data/targets; size *output\\_dim* × *N\\_samples*
  - `obs_noise_cov` - covariance of observational noise
  - `truncate_svd` - Project onto this fraction of the largest principal components. Defaults
    to 1.0 (no truncation).

Returns the transformed data and the decomposition, which is a matrix 
factorization of type LinearAlgebra.SVD. 
  
Note: If `F::SVD` is the factorization object, U, S, V and Vt can be obtained via 
F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt. The singular values 
in S are sorted in descending order.
"""
function svd_transform(
    data::AbstractMatrix{FT},
    obs_noise_cov::Union{AbstractMatrix{FT}, Nothing};
    retained_svd_frac::FT = 1.0,
) where {FT <: AbstractFloat}
    if obs_noise_cov === nothing
        # no-op case
        return data, nothing
    end
    # actually have a matrix to take the SVD of
    decomp = svd(obs_noise_cov)
    sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomp.S))
    if retained_svd_frac < 1.0
        # Truncate the SVD as a form of regularization
        # Find cutoff
        S_cumsum = cumsum(decomp.S) / sum(decomp.S)
        ind = findall(x -> (x > retained_svd_frac), S_cumsum)
        k = ind[1]
        n = size(data)[1]
        println("SVD truncated at k: ", k, "/", n)
        # Apply truncated SVD
        transformed_data = sqrt_singular_values_inv[1:k, 1:k] * decomp.Vt[1:k, :] * data
        decomposition = SVD(decomp.U[:, 1:k], decomp.S[1:k], decomp.Vt[1:k, :])
    else
        transformed_data = sqrt_singular_values_inv * decomp.Vt * data
        decomposition = svd(obs_noise_cov)
    end
    return transformed_data, decomposition
end

function svd_transform(
    data::AbstractVector{FT},
    obs_noise_cov::Union{AbstractMatrix{FT}, Nothing};
    retained_svd_frac::FT = 1.0,
) where {FT <: AbstractFloat}
    # method for 1D data
    transformed_data, decomposition =
        svd_transform(reshape(data, :, 1), obs_noise_cov; retained_svd_frac = retained_svd_frac)
    return vec(transformed_data), decomposition
end

function svd_transform(
    data::AbstractVecOrMat{FT},
    obs_noise_cov::UniformScaling{FT};
    retained_svd_frac::FT = 1.0,
) where {FT <: AbstractFloat}
    # method for UniformScaling
    return svd_transform(data, Diagonal(obs_noise_cov, size(data, 1)); retained_svd_frac = retained_svd_frac)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Transform the mean and covariance back to the original (correlated) coordinate system
  - `μ` - predicted mean; size *output\\_dim* × *N\\_predicted\\_points*.
  - `σ2` - predicted variance; size *output\\_dim* × *N\\_predicted\\_points*.
  - `decomposition` - SVD decomposition of *obs\\_noise\\_cov*.

Returns the transformed mean (size *output\\_dim* × *N\\_predicted\\_points*) and variance. 
Note that transforming the variance back to the original coordinate system
results in non-zero off-diagonal elements, so instead of just returning the 
elements on the main diagonal (i.e., the variances), we return the full 
covariance at each point, as a vector of length *N\\_predicted\\_points*, where 
each element is a matrix of size *output\\_dim* × *output\\_dim*.
"""
function svd_reverse_transform_mean_cov(
    μ::AbstractMatrix{FT},
    σ2::AbstractMatrix{FT},
    decomposition::SVD,
) where {FT <: AbstractFloat}
    @assert size(μ) == size(σ2)

    output_dim, N_predicted_points = size(σ2)
    # We created meanvGP = D_inv * Vt * mean_v so meanv = V * D * meanvGP
    sqrt_singular_values = Diagonal(sqrt.(decomposition.S))
    transformed_μ = decomposition.V * sqrt_singular_values * μ

    transformed_σ2 = Vector{Symmetric{FT, Matrix{FT}}}(undef, N_predicted_points)
    # Back transformation

    for j in 1:N_predicted_points
        σ2_j = decomposition.V * sqrt_singular_values * Diagonal(σ2[:, j]) * sqrt_singular_values * decomposition.Vt
        transformed_σ2[j] = Symmetric(σ2_j)
    end

    return transformed_μ, transformed_σ2
end



end
