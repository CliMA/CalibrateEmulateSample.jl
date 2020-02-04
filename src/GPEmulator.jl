"""
    GPEmulator

To create an emulator (GP).
Include functions to optimize the emulator
and to make predictions of mean and variance
uses ScikitLearn (or GaussianProcesses.jl
"""
module GPEmulator

using ..EKI
using ..Truth
const S = EKI.EKIObj
const T = Truth.TruthObj

# packages
using Statistics
using Distributions
using LinearAlgebra
using GaussianProcesses
using ScikitLearn
using Optim

@sk_import gaussian_process : GaussianProcessRegressor
@sk_import gaussian_process.kernels : (RBF, WhiteKernel, ConstantKernel)
const sk = ScikitLearn
#using Interpolations #will use this later (faster)
#using PyCall
#py=pyimport("scipy.interpolate")

# exports
export GPObj
export predict
export emulate
export extract

#####
##### Structure definitions
#####

#structure to hold inputs/ouputs kernel type and whether we do sparse GP
struct GPObj{FT<:AbstractFloat}
    inputs::Matrix{FT}
    data::Matrix{FT}
    models::Vector
    package::String
end

function GPObj(inputs, data, package)
    FT = eltype(data)
    if package == "gp_jl"
        models = Any[]
        outputs = convert(Matrix{FT}, data')
        inputs = convert(Matrix{FT}, inputs')

        for i in 1:size(outputs, 1)
            # Zero mean function
            kmean = MeanZero()
            # Construct kernel:
            # Sum kernel consisting of Matern 5/2 ARD kernel and Squared
            # Exponential kernel
            len2 = FT(1)
            var2 = FT(1)
            kern1 = SE(len2, var2)
            kern2 = Matern(5/2, FT[0.0, 0.0, 0.0], FT(0.0))
            lognoise = FT(0.5)
            # regularize with white noise
            white = Noise(log(FT(2.0)))
            # construct kernel
            kern = kern1 + kern2 + white

            # inputs: N_param x N_samples
            # outputs: N_data x N_samples
            m = GPE(inputs, outputs[i, :], kmean, kern, sqrt(lognoise))
            optimize!(m)
            push!(models, m)
        end

    elseif package == "sk_jl"

        len2 = ones(size(inputs, 2))
        var2 = FT(1)

        varkern = ConstantKernel(constant_value=var2,
                                 constant_value_bounds=(FT(1e-05), FT(1000.0)))
        rbf = RBF(length_scale=len2, length_scale_bounds=(FT(1), FT(1000.0)))

        white = WhiteKernel(noise_level=1.0, noise_level_bounds=(FT(1e-05), FT(10.0)))
        kern = varkern * rbf + white
        models = Any[]

        outputs = convert(Matrix{FT}, data')
        inputs = convert(Matrix{FT}, inputs)

        for i in 1:size(outputs,1)
            out = reshape(outputs[i,:], (size(outputs, 2), 1))

            m = GaussianProcessRegressor(kernel=kern,
                                         n_restarts_optimizer=10,
                                         alpha=FT(0.0), normalize_y=true)
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
        new_inputs = convert(Matrix{FT}, new_inputs')
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

end # function predict


function emulate(u_tp::Array{FT, 2}, g_tp::Array{FT, 2},
                 gppackage::String) where {FT<:AbstractFloat}

    gpobj = GPObj(u_tp, g_tp, gppackage) # construct the GP based on data

    return gpobj
end


function extract(truthobj::T, ekiobj::EKIObj{FT}, N_eki_it::I) where {T,FT, I<:Int}

    yt = truthobj.y_t
    yt_cov = truthobj.cov
    yt_covinv = inv(yt_cov)

    # Note u[end] does not have an equivalent g
    u_tp = ekiobj.u[end-N_eki_it:end-1] # N_eki_it x [N_ens x N_param]
    g_tp = ekiobj.g[end-N_eki_it+1:end] # N_eki_it x [N_ens x N_data]

    # u does not require reduction, g does:
    # g_tp[j] is jth iteration of ensembles
    u_tp = cat(u_tp..., dims=1) # [(N_eki_it x N_ens) x N_param]
    g_tp = cat(g_tp..., dims=1) # [(N_eki_it x N_ens) x N_data]

    return yt, yt_cov, yt_covinv, u_tp, g_tp
end

function orig2zscore(X::AbstractVector{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    # Compute the z scores of a vector X using the given mean
    # and std
    Z = zeros(size(X))
    for i in 1:length(X)
        Z[i] = (X[i] - mean[i]) / std[i]
    end
    return Z
end

function orig2zscore(X::AbstractMatrix{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    # Compute the z scores of matrix X using the given mean and
    # std. Transformation is applied column-wise.
    s = size(X)
    Z = zeros(s)
    n_cols = s[2]
    for i in 1:n_cols
        Z[:,i] = (X[:,i] .- mean[i]) ./ std[i]
    end
    return Z
end

function zscore2orig(Z::AbstractVector{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    # Transform X (a vector of z scores) back to the original
    # values
    X = zeros(size(Z))
    for i in 1:length(X)
      X[i] = Z[i] .* std[i] .+ mean[i]
    end
    return X
end

function zscore2orig(Z::AbstractMatrix{FT},
                     mean::AbstractVector{FT},
                     std::AbstractVector{FT}) where {FT}
    s = size(Z)
    X = zeros(s)
    # Transform X (a matrix of z scores) back to the original
    # values. Transformation is applied column-wise.
    n_cols = s[2]
    for i in 1:n_cols
        X[:,i] = Z[:,i] .* std[i] .+ mean[i]
    end
    return X
end

end # module GPEmulator
