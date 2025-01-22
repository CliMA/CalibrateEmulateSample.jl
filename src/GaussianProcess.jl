
using EnsembleKalmanProcesses.DataContainers
using DocStringExtensions

# [1] For GaussianProcesses
import GaussianProcesses: predict, get_params, get_param_names
using GaussianProcesses

# [2] For SciKitLearn
using PyCall
using ScikitLearn
const pykernels = PyNULL()
const pyGP = PyNULL()
function __init__()
    copy!(pykernels, pyimport_conda("sklearn.gaussian_process.kernels", "scikit-learn=1.5.1"))
    copy!(pyGP, pyimport_conda("sklearn.gaussian_process", "scikit-learn=1.5.1"))
end

# [3] For AbstractGPs
using AbstractGPs
using KernelFunctions

#exports (from Emulator)
export GaussianProcess

export GPJL, SKLJL, AGPJL
export YType, FType

export get_params
export get_param_names


"""
$(DocStringExtensions.TYPEDEF)

Type to dispatch which GP package to use:

 - `GPJL` for GaussianProcesses.jl, [julia - gradient-free only]
 - `SKLJL` for the ScikitLearn GaussianProcessRegressor, [python - gradient-free]
 - `AGPJL` for AbstractGPs.jl, [julia - ForwardDiff compatible]
"""
abstract type GaussianProcessesPackage end
struct GPJL <: GaussianProcessesPackage end
struct SKLJL <: GaussianProcessesPackage end
struct AGPJL <: GaussianProcessesPackage end

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
    models::Vector{Union{<:GaussianProcesses.GPE, <:PyObject, <:AbstractGPs.PosteriorGP, Nothing}}
    "Kernel object."
    kernel::Union{<:GaussianProcesses.Kernel, <:PyObject, <:AbstractGPs.Kernel, Nothing}
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
    kernel::Union{K, KPy, AGPK, Nothing} = nothing,
    noise_learn = true,
    alg_reg_noise::FT = 1e-3,
    prediction_type::PredictionType = YType(),
) where {
    GPPkg <: GaussianProcessesPackage,
    K <: GaussianProcesses.Kernel,
    KPy <: PyObject,
    AGPK <: AbstractGPs.Kernel,
    FT <: AbstractFloat,
}

    # Initialize vector for GP models
    models = Vector{Union{<:GaussianProcesses.GPE, <:PyObject, <:AbstractGPs.PosteriorGP, <:Nothing}}(undef, 0)

    # the algorithm regularization noise is set to some small value if we are learning noise, else
    # it is fixed to the correct value (1.0)
    if !(noise_learn)
        alg_reg_noise = 1.0
    end

    return GaussianProcess{typeof(package), FT}(models, kernel, noise_learn, alg_reg_noise, prediction_type)
end

# First we create  the GPJL implementation
"""
Gets flattened kernel hyperparameters from a (vector of) `GaussianProcess{GPJL}` model(s). Extends GaussianProcess.jl method.
"""
function GaussianProcesses.get_params(gp::GaussianProcess{GPJL})
    return [get_params(model.kernel) for model in gp.models]
end

"""
Gets the flattened names of kernel hyperparameters from a (vector of) `GaussianProcess{GPJL}` model(s). Extends GaussianProcess.jl method.
"""
function GaussianProcesses.get_param_names(gp::GaussianProcess{GPJL})
    return [get_param_names(model.kernel) for model in gp.models]
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Method to build Gaussian process models based on the package.
"""
function build_models!(
    gp::GaussianProcess{GPJL},
    input_output_pairs::PairedDataContainer{FT};
    kwargs...,
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
    input_output_pairs::PairedDataContainer{FT};
    kwargs...,
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

        @info("Training kernel $(i), ")
        ScikitLearn.fit!(m, input_values, data_i)
        push!(models, m)
        @info(m.kernel)
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


#We build the AGPJL implementation
function build_models!(
    gp::GaussianProcess{AGPJL},
    input_output_pairs::PairedDataContainer{FT};
    kernel_params = nothing,
    kwargs...,
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

    ##############################################################################
    # Notes on borrowing hyperparameters optimised within GPJL:
    # optimisation of the GPJL with default kernels produces kernel parameters
    # in the way of [a b c], where:
    # c is the log_const_value
    # [a b] is the rbf_len: lengthscale parameters for SEArd kernel
    # const_value = exp.(2 .* log_const_value)
    ##############################################################################
    ## For example A 2D->2D sinusoid input example:
    #=
    log_const_value = [2.9031145778344696; 3.8325906110973795]
    rbf_len = [1.9952706691900783 3.066374123568536; 5.783676639895112 2.195849064147456]
    =#
    if isnothing(kernel_params)
        throw(ArgumentError("""
AbstractGP currently does not (yet) learn hyperparameters internally. The following can be performed instead:
1. Create and optimize a GPJL emulator and default kernel. (here called gp_jl)
2. Create the Kernel parameters as a vect-of-dict with
   kernel_params = [
       Dict(
           "log_rbf_len" => model_params[1:end-2] # input-dim Vector,
           "log_std_sqexp" => model_params[end-2] # Float,
           "log_std_noise" => # Float,
       )
    for model_params in get_params(gp_jl)]
    Note: get_params(gp_jl) returns `output_dim`-vector where each entry is [a, b, c] with:
    - a is the `rbf_len`: lengthscale parameters for SEArd kernel [input_dim] Vector
    - b is the `log_std_sqexp` of the SQexp kernel Float
    - c is the `log_std_noise` of the noise kernel Float
3. Build a new Emulator with kwargs `kernel_params=kernel_params`
        """))
    end


    N_models = size(output_values, 1) #size(transformed_data)[1]
    regularization_noise = gp.alg_reg_noise
    
    # now obtain the values of the hyperparameters
    if N_models == 1 && !(isa(kernel_params, AbstractVector)) # i.e. just a Dict
        kernel_params_vec = [kernel_params]
    else
        kernel_params_vec = kernel_params
    end

    for i in 1:N_models
        var_sqexp = exp.(2 .* kernel_params_vec[i]["log_std_sqexp"]) # Float
        var_noise = exp.(2 .* kernel_params_vec[i]["log_std_noise"]) # Float
        rbf_invlen = 1 ./ exp.(kernel_params_vec[i]["log_rbf_len"])# Vec

        opt_kern =
            var_sqexp * (KernelFunctions.SqExponentialKernel() ∘ ARDTransform(rbf_invlen[:])) +
            var_noise * KernelFunctions.WhiteKernel()
        opt_f = AbstractGPs.GP(opt_kern)
        opt_fx = opt_f(input_values', regularization_noise)

        data_i = output_values[i, :]
        opt_post_fx = posterior(opt_fx, data_i)
        println("optimised GP: ", i)
        push!(models, opt_post_fx)
        println(opt_post_fx.prior.kernel)
    end

end

function optimize_hyperparameters!(gp::GaussianProcess{AGPJL}, args...; kwargs...)
    @info "AbstractGP already built. Continuing..."
end

function predict(gp::GaussianProcess{AGPJL}, new_inputs::AM) where {AM <: AbstractMatrix}

    N_models = length(gp.models)
    N_samples = size(new_inputs, 2)
    FTorD = eltype(new_inputs) # e.g. Float or Dual
    μ = zeros(FTorD, N_models, N_samples)
    σ2 = zeros(FTorD, N_models, N_samples)
    for i in 1:N_models
        pred_gp = gp.models[i]
        pred = pred_gp(new_inputs)
        μ[i, :] = mean(pred)
        σ2[i, :] = var(pred)
    end

    σ2[:, :] .= σ2[:, :] .+ gp.alg_reg_noise
    return μ, σ2
end
