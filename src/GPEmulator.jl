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
export emulate
export extract


"""
  GPObj

  Structure holding training input and the fitted Gaussian Process Regression
  models.

  Attributes
  - `inputs` - array of training points/parameters; N_parameters x N_samples
  - `outputs` - array of corresponding data ("outputs = G(inputs)");
                N_parameters x N_samples
  - `models` - the Gaussian Process Regression model(s) that are fitted to
               the given input-output pairs
  - `package` - which GP package to use - "gp_jl" for GaussianProcesses.jl,
                or "sk_jl" for the ScikitLearn GaussianProcessRegressor
  - `GPkernel` - GaussianProcesses kernel object. If not supplied, a default
                 kernel is used. The default kernel is the sum of a Squared
                 Exponential kernel and white noise.

  # Constructors
    GPObj(inputs, data, package, [GPkernel]), with inputs and data of size
    N_samples x N_parameters (both arrays will be transposed in the
    construction of the GPObj)

"""
struct GPObj{FT<:AbstractFloat}
    inputs::Array{FT, 2}
    data::Array{FT, 2}
    models::Vector
    package::String
end

function GPObj(inputs, data, package, GPkernel::Union{K, KPy, Nothing}=nothing) where {K<:Kernel, KPy<:PyObject}
    FT = eltype(data)
    if package == "gp_jl"
        models = Any[]
        outputs = convert(Array{FT}, data')
        inputs = convert(Array{FT}, inputs')

        # Use a default kernel unless a kernel was supplied to GPObj
        if typeof(GPkernel)==Nothing
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

    elseif package == "sk_jl"
        models = Any[]
        outputs = convert(Array{FT}, data')
        inputs = convert(Array{FT}, inputs)

        if typeof(GPkernel)==Nothing
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

    else
        error("use package sk_jl or gp_jl")
    end

    return GPObj{FT}(inputs, outputs, models, package)
end


function predict(gp::GPObj{FT}, new_inputs::Array{FT};
                 prediction_type="y") where {FT}

    # predict data (type "y") or latent function (type "f")
    # column of new_inputs gives new parameter set to evaluate gp at
    M = length(gp.models)
    mean = Array{FT}[]
    var = Array{FT}[]
    # predicts columns of inputs so must be transposed
    if gp.package == "gp_jl"
        new_inputs = convert(Array{FT}, new_inputs')
    end
    for i=1:M
        if gp.package == "gp_jl"
            if prediction_type == "y"
                mu, sig2 = predict_y(gp.models[i], new_inputs)
                push!(mean, mu)
                push!(var, sig2)

            elseif prediction_type == "f"
                mu, sig2 = predict_f(gp.models[i], new_inputs')
                push!(mean, mu)
                push!(var, sig2)

            else
                println("prediction_type must be string: y or f")
                exit()
            end

        elseif gp.package == "sk_jl"
            mu, sig = gp.models[i].predict(new_inputs, return_std=true)
            sig2 = sig .* sig
            push!(mean, mu)
            push!(var, sig2)
        end
    end

    return mean, var

end

end # module GPEmulator
