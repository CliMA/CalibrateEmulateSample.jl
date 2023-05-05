
using RandomFeatures
const RF = RandomFeatures
using EnsembleKalmanProcesses
const EKP = EnsembleKalmanProcesses
using ..ParameterDistributions
using ..Utilities
using StableRNGs
using Random

abstract type RandomFeatureInterface <: MachineLearningTool end

abstract type MultithreadType end
struct TullioThreading <: MultithreadType end
struct EnsembleThreading <: MultithreadType end

#common functions
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Converts a flat array into a cholesky factor.
"""
function flat_to_chol(x::V) where {V <: AbstractVector}
    choldim = Int(floor(sqrt(2 * length(x))))
    cholmat = zeros(choldim, choldim)
    for i in 1:choldim
        for j in 1:i
            cholmat[i, j] = x[sum(0:(i - 1)) + j]
        end
    end
    return cholmat
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds Covariance matrices from a flat array of hyperparameters
"""
function hyperparameters_from_flat(
    x::VV,
    input_dim::Int,
    output_dim::Int,
    diagonalize_input::Bool,
    diagonalize_output::Bool,
) where {VV <: AbstractVector}
    # if dim = 1 then parameters are a 1x1 matrix
    # if we diagonalize then parameters are diagonal entries + constant
    # if we don't diagonalize then parameters are a cholesky product + constant on diagonal.

    #input space
    if input_dim == 1
        in_max = 1
        U = x[in_max] * ones(1, 1)
    elseif diagonalize_input
        in_max = input_dim + 2
        U = convert(Matrix, x[in_max - 1] * (Diagonal(x[1:(in_max - 2)]) + x[in_max] * I))
    elseif !diagonalize_input
        in_max = Int(input_dim * (input_dim + 1) / 2) + 2
        cholU = flat_to_chol(x[1:(in_max - 2)])
        U = x[in_max - 1] * (cholU * permutedims(cholU, (2, 1)) + x[in_max] * I)
    end

    # output_space    
    out_min = in_max + 1
    if output_dim == 1
        V = x[end] * ones(1, 1)
    elseif diagonalize_output
        V = convert(Matrix, x[end - 1] * (Diagonal(x[out_min:(end - 2)]) + x[end] * I))
    elseif !diagonalize_output
        cholV = flat_to_chol(x[out_min:(end - 2)])
        V = x[end - 1] * (cholV * permutedims(cholV, (2, 1)) + x[end] * I)
    end

    # sometimes this process does not give a (numerically) symmetric matrix
    U = 0.5 * (U + permutedims(U, (2, 1)))
    V = 0.5 * (V + permutedims(V, (2, 1)))

    return U, V

end

function hyperparameters_from_flat(x::VV, input_dim::Int, diagonalize_input::Bool) where {VV <: AbstractVector}
    output_dim = 1
    diagonalize_output = false
    U, V = hyperparameters_from_flat(x, input_dim, output_dim, diagonalize_input, diagonalize_output)
    # Note that the scalar setting has a slight inconsistency from the 1-D output vector case
    # We correct here by rescaling U using the single hyperparameter in V
    return V[1, 1] * U
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds a prior over the hyperparameters (i.e. the cholesky/diagaonal or individaul entries of the input/output covariances).
For example, the case where the input covariance ``U = γ_1 * (LL^T + γ_2 I)``, 
we set priors for the entries of the lower triangular matrix  ``L`` as normal, and constant scalings ``γ_i`` as log-normal to retain positivity.
"""
function build_default_prior(input_dim::Int, output_dim::Int, diagonalize_input::Bool, diagonalize_output::Bool)
    n_hp_in = n_hyperparameters_cov(input_dim, diagonalize_input)
    n_hp_out = n_hyperparameters_cov(output_dim, diagonalize_output)

    # if dim = 1 , positive constant
    # elseif diagonalized, positive on diagonal and positive scalings
    # else chol factor, and positive scalings
    if input_dim > 1
        if diagonalize_input
            param_diag = constrained_gaussian("input_prior_diagonal", 1.0, 1.0, 0.0, Inf, repeats = n_hp_in - 2)
            param_scaling = constrained_gaussian("input_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2)
            input_prior = combine_distributions([param_diag, param_scaling])
        else
            param_chol = constrained_gaussian("input_prior_cholesky", 0.0, 1.0, -Inf, Inf, repeats = n_hp_in - 2)
            param_scaling = constrained_gaussian("input_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2)
            input_prior = combine_distributions([param_chol, param_scaling])
        end
    else
        input_prior = constrained_gaussian("input_prior", 1.0, 1.0, 0.0, Inf)
    end

    if output_dim > 1
        if diagonalize_output
            param_diag = constrained_gaussian("output_prior_diagonal", 1.0, 1.0, 0.0, Inf, repeats = n_hp_out - 2)
            param_scaling = constrained_gaussian("output_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2)
            output_prior = combine_distributions([param_diag, param_scaling])
        else
            param_chol = constrained_gaussian("output_prior_cholesky", 0.0, 1.0, -Inf, Inf, repeats = n_hp_out - 2)
            param_scaling = constrained_gaussian("output_prior_scaling", 1.0, 1.0, 0.0, Inf, repeats = 2)
            output_prior = combine_distributions([param_chol, param_scaling])
        end
    else
        output_prior = constrained_gaussian("output_prior", 1.0, 1.0, 0.0, Inf)
    end

    return combine_distributions([input_prior, output_prior])

end

function build_default_prior(input_dim::Int, diagonalize_input::Bool)
    #scalar case consistent with vector case
    output_dim = 1
    diagonalize_output = false
    return build_default_prior(input_dim, output_dim, diagonalize_input, diagonalize_output)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

The number of hyperparameters required to characterize covariance (determined from `hyperparameters_from_flat`).
"""
function n_hyperparameters_cov(dim::Int, diagonalize::Bool)
    n_hp = 0
    n_hp += diagonalize ? dim : Int(dim * (dim + 1) / 2)     # inputs: diagonal or cholesky
    n_hp += (dim > 1) ? 2 : 0     # add two more for constant diagonal in each case.
    return n_hp
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Calculate number of hyperparameters required to create the default prior in the given input/output space (determined from `hyperparameters_from_flat`)
"""
function calculate_n_hyperparameters(input_dim::Int, output_dim::Int, diagonalize_input::Bool, diagonalize_output::Bool)
    n_hp_in = n_hyperparameters_cov(input_dim, diagonalize_input)
    n_hp_out = n_hyperparameters_cov(output_dim, diagonalize_output)
    return n_hp_in + n_hp_out
end

function calculate_n_hyperparameters(input_dim::Int, diagonalize_input::Bool)
    #scalar case consistent with vector case
    output_dim = 1
    diagonalize_output = false
    return calculate_n_hyperparameters(input_dim, output_dim, diagonalize_input, diagonalize_output)
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Calculates key quantities (mean, covariance, and coefficients) of the objective function to be optimized for hyperparameter training. Buffers: mean_store, cov_store, buffer are provided to aid memory management.
"""
function calculate_mean_cov_and_coeffs(
    rfi::RFI,
    rng::RNG,
    l::ForVM,
    regularization::MorUSorD,
    n_features::Int,
    n_train::Int,
    n_test::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    io_pairs::PairedDataContainer,
    decomp_type::S,
    mean_store::M,
    cov_store::A,
    buffer::A,
    multithread_type::MT,
) where {
    RFI <: RandomFeatureInterface,
    RNG <: AbstractRNG,
    ForVM <: Union{AbstractFloat, AbstractVecOrMat},
    S <: AbstractString,
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
    M <: AbstractMatrix{<:AbstractFloat},
    A <: AbstractArray{<:AbstractFloat, 3},
    MT <: MultithreadType,
}

    # split data into train/test 
    itrain = get_inputs(io_pairs)[:, 1:n_train]
    otrain = get_outputs(io_pairs)[:, 1:n_train]
    io_train_cost = PairedDataContainer(itrain, otrain)
    itest = get_inputs(io_pairs)[:, (n_train + 1):end]
    otest = get_outputs(io_pairs)[:, (n_train + 1):end]
    input_dim = size(itrain, 1)
    output_dim = size(otrain, 1)

    # build and fit the RF
    rfm = RFM_from_hyperparameters(
        rfi,
        rng,
        l,
        regularization,
        n_features,
        batch_sizes,
        input_dim,
        output_dim,
        multithread_type,
    )
    fitted_features = RF.Methods.fit(rfm, io_train_cost, decomposition_type = decomp_type)

    #we want to calc 1/var(y-mean)^2 + lambda/m * coeffs^2 in the end
    thread_opt = isa(multithread_type, TullioThreading)
    RF.Methods.predict!(
        rfm,
        fitted_features,
        DataContainer(itest),
        mean_store,
        cov_store,
        buffer,
        tullio_threading = thread_opt,
    )
    # sizes (output_dim x n_test), (output_dim x output_dim x n_test) 
    scaled_coeffs = sqrt(1 / n_features) * RF.Methods.get_coeffs(fitted_features)

    # we add the additional complexity term
    # TODO only cholesky and SVD available
    if decomp_type == "cholesky"
        chol_fac = RF.Methods.get_decomposition(RF.Methods.get_feature_factors(fitted_features)).L
        complexity = 2 * sum(log(chol_fac[i, i]) for i in 1:size(chol_fac, 1))
    else
        svd_singval = RF.Methods.get_decomposition(RF.Methods.get_feature_factors(fitted_features)).S
        complexity = sum(log, svd_singval) # note this is log(abs(det))
    end
    return scaled_coeffs, complexity

end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Estimation of objective function (marginal log-likelihood) variability for optimization of hyperparameters. Calculated empirically by calling many samples of the objective at a "reasonable" value. `multithread_type` is used as dispatch, either `ensemble` (thread over the empirical samples) or `tullio` (serial samples, and thread the linear algebra within each sample)
"""
function estimate_mean_and_coeffnorm_covariance(
    rfi::RFI,
    rng::RNG,
    l::ForVM,
    regularization::MorUSorD,
    n_features::Int,
    n_train::Int,
    n_test::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    io_pairs::PairedDataContainer,
    n_samples::Int,
    decomp_type::S,
    multithread_type::TullioThreading;
    repeats::Int = 1,
) where {
    RFI <: RandomFeatureInterface,
    RNG <: AbstractRNG,
    ForVM <: Union{AbstractFloat, AbstractVecOrMat},
    S <: AbstractString,
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
}

    output_dim = size(get_outputs(io_pairs), 1)

    means = zeros(output_dim, n_samples, n_test)
    mean_of_covs = zeros(output_dim, output_dim, n_test)
    moc_tmp = similar(mean_of_covs)
    mtmp = zeros(output_dim, n_test)
    buffer = zeros(n_test, output_dim, n_features)
    complexity = zeros(1, n_samples)
    coeffl2norm = zeros(1, n_samples)
    println("estimate cov with " * string(n_samples * repeats) * " iterations...")

    for i in ProgressBar(1:n_samples)
        for j in 1:repeats
            c, cplxty = calculate_mean_cov_and_coeffs(
                rfi,
                rng,
                l,
                regularization,
                n_features,
                n_train,
                n_test,
                batch_sizes,
                io_pairs,
                decomp_type,
                mtmp,
                moc_tmp,
                buffer,
                multithread_type,
            )

            # m output_dim x n_test
            # v output_dim x output_dim x n_test
            # c n_features
            # cplxty 1

            # update vbles needed for cov
            means[:, i, :] .+= mtmp ./ repeats
            coeffl2norm[1, i] += sqrt(sum(abs2, c)) / repeats
            complexity[1, i] += cplxty / repeats

            # update vbles needed for mean
            @. mean_of_covs += moc_tmp / (repeats * n_samples)

        end
    end
    means = permutedims(means, (3, 2, 1))
    mean_of_covs = permutedims(mean_of_covs, (3, 1, 2))

    approx_σ2 = zeros(n_test * output_dim, n_test * output_dim)
    blockmeans = zeros(n_test * output_dim, n_samples)
    for i in 1:n_test
        id = ((i - 1) * output_dim + 1):(i * output_dim)
        approx_σ2[id, id] = mean_of_covs[i, :, :] # this ordering, so we can take a mean/cov in dims = 2.
        blockmeans[id, :] = permutedims(means[i, :, :], (2, 1))
    end

    if !isposdef(approx_σ2)
        println("approx_σ2 not posdef")
        approx_σ2 = posdef_correct(approx_σ2)
    end

    Γ = cov(vcat(blockmeans, coeffl2norm, complexity), dims = 2)


    return Γ, approx_σ2

end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluate an objective function (marginal log-likelihood) over an ensemble of hyperparameter values (used for EKI). `multithread_type` is used as dispatch, either `ensemble` (thread over the ensemble members) or `tullio` (serial samples, and thread the linear algebra within each sample)
"""
function calculate_ensemble_mean_and_coeffnorm(
    rfi::RFI,
    rng::RNG,
    lvecormat::VorM,
    regularization::MorUSorD,
    n_features::Int,
    n_train::Int,
    n_test::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    io_pairs::PairedDataContainer,
    decomp_type::S,
    multithread_type::TullioThreading;
    repeats::Int = 1,
) where {
    RFI <: RandomFeatureInterface,
    RNG <: AbstractRNG,
    VorM <: AbstractVecOrMat,
    S <: AbstractString,
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
}
    if isa(lvecormat, AbstractVector)
        lmat = reshape(lvecormat, 1, :)
    else
        lmat = lvecormat
    end
    N_ens = size(lmat, 2)
    output_dim = size(get_outputs(io_pairs), 1)

    means = zeros(output_dim, N_ens, n_test)
    mean_of_covs = zeros(output_dim, output_dim, n_test)
    buffer = zeros(n_test, output_dim, n_features)
    complexity = zeros(1, N_ens)
    coeffl2norm = zeros(1, N_ens)
    moc_tmp = similar(mean_of_covs)
    mtmp = zeros(output_dim, n_test)

    println("calculating " * string(N_ens * repeats) * " ensemble members...")

    for i in ProgressBar(1:N_ens)
        for j in collect(1:repeats)
            l = lmat[:, i]

            c, cplxty = calculate_mean_cov_and_coeffs(
                rfi,
                rng,
                l,
                regularization,
                n_features,
                n_train,
                n_test,
                batch_sizes,
                io_pairs,
                decomp_type,
                mtmp,
                moc_tmp,
                buffer,
                multithread_type,
            )
            # m output_dim x n_test
            # v output_dim x output_dim x n_test
            # c n_features
            means[:, i, :] += mtmp ./ repeats
            @. mean_of_covs += moc_tmp / (repeats * N_ens)
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
            complexity[1, i] += cplxty / repeats
        end
    end
    means = permutedims(means, (3, 2, 1))
    mean_of_covs = permutedims(mean_of_covs, (3, 1, 2))
    blockcovmat = zeros(n_test * output_dim, n_test * output_dim)
    blockmeans = zeros(n_test * output_dim, N_ens)
    for i in 1:n_test
        id = ((i - 1) * output_dim + 1):(i * output_dim)
        blockcovmat[id, id] = mean_of_covs[i, :, :]
        blockmeans[id, :] = permutedims(means[i, :, :], (2, 1))
    end

    if !isposdef(blockcovmat)
        println("blockcovmat not posdef")
        blockcovmat = posdef_correct(blockcovmat)
    end

    return vcat(blockmeans, coeffl2norm, complexity), blockcovmat

end


function estimate_mean_and_coeffnorm_covariance(
    rfi::RFI,
    rng::RNG,
    l::ForVM,
    regularization::MorUSorD,
    n_features::Int,
    n_train::Int,
    n_test::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    io_pairs::PairedDataContainer,
    n_samples::Int,
    decomp_type::S,
    multithread_type::EnsembleThreading;
    repeats::Int = 1,
) where {
    RFI <: RandomFeatureInterface,
    RNG <: AbstractRNG,
    ForVM <: Union{AbstractFloat, AbstractVecOrMat},
    S <: AbstractString,
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
}

    output_dim = size(get_outputs(io_pairs), 1)

    println("estimate cov with " * string(n_samples * repeats) * " iterations...")

    nthreads = Threads.nthreads()
    rng_seed = randperm(rng, 10^5)[1] # dumb way to get a random integer in 1:10^5
    rng_list = [Random.MersenneTwister(rng_seed + i) for i in 1:nthreads]

    c_list = [zeros(1, n_samples) for i in 1:nthreads]
    cp_list = [zeros(1, n_samples) for i in 1:nthreads]
    m_list = [zeros(output_dim, n_samples, n_test) for i in 1:nthreads]
    moc_list = [zeros(output_dim, output_dim, n_test) for i in 1:nthreads]
    moc_tmp_list = [zeros(output_dim, output_dim, n_test) for i in 1:nthreads]
    mtmp_list = [zeros(output_dim, n_test) for i in 1:nthreads]
    buffer_list = [zeros(n_test, output_dim, n_features) for i in 1:nthreads]


    Threads.@threads for i in ProgressBar(1:n_samples)
        tid = Threads.threadid()
        rngtmp = rng_list[tid]
        mtmp = mtmp_list[tid]
        moc_tmp = moc_tmp_list[tid]
        buffer = buffer_list[tid]
        for j in 1:repeats

            c, cplxty = calculate_mean_cov_and_coeffs(
                rfi,
                rngtmp,
                l,
                regularization,
                n_features,
                n_train,
                n_test,
                batch_sizes,
                io_pairs,
                decomp_type,
                mtmp,
                moc_tmp,
                buffer,
                multithread_type,
            )


            # m output_dim x n_test
            # v output_dim x output_dim x n_test
            # c n_features
            # cplxty 1

            # update vbles needed for cov
            # update vbles needed for cov
            m_list[tid][:, i, :] += mtmp ./ repeats
            c_list[tid][1, i] += sqrt(sum(abs2, c)) / repeats
            cp_list[tid][1, i] += cplxty / repeats

            # update vbles needed for mean
            @. moc_list[tid] += moc_tmp ./ (repeats * n_samples)

        end
    end

    #put back together after threading
    mean_of_covs = sum(moc_list)
    means = sum(m_list)
    coeffl2norm = sum(c_list)
    complexity = sum(cp_list)
    means = permutedims(means, (3, 2, 1))
    mean_of_covs = permutedims(mean_of_covs, (3, 1, 2))

    approx_σ2 = zeros(n_test * output_dim, n_test * output_dim)
    blockmeans = zeros(n_test * output_dim, n_samples)
    for i in 1:n_test
        id = ((i - 1) * output_dim + 1):(i * output_dim)
        approx_σ2[id, id] = mean_of_covs[i, :, :] # this ordering, so we can take a mean/cov in dims = 2.
        blockmeans[id, :] = permutedims(means[i, :, :], (2, 1))
    end

    if !isposdef(approx_σ2)
        println("approx_σ2 not posdef")
        approx_σ2 = posdef_correct(approx_σ2)
    end

    Γ = cov(vcat(blockmeans, coeffl2norm, complexity), dims = 2)
    return Γ, approx_σ2

end

function calculate_ensemble_mean_and_coeffnorm(
    rfi::RFI,
    rng::RNG,
    lvecormat::VorM,
    regularization::MorUSorD,
    n_features::Int,
    n_train::Int,
    n_test::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    io_pairs::PairedDataContainer,
    decomp_type::S,
    multithread_type::EnsembleThreading;
    repeats::Int = 1,
) where {
    RFI <: RandomFeatureInterface,
    RNG <: AbstractRNG,
    VorM <: AbstractVecOrMat,
    S <: AbstractString,
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
}
    if isa(lvecormat, AbstractVector)
        lmat = reshape(lvecormat, 1, :)
    else
        lmat = lvecormat
    end
    N_ens = size(lmat, 2)
    output_dim = size(get_outputs(io_pairs), 1)



    println("calculating " * string(N_ens * repeats) * " ensemble members...")

    nthreads = Threads.nthreads()

    means = zeros(output_dim, N_ens, n_test)
    complexity = zeros(1, N_ens)
    coeffl2norm = zeros(1, N_ens)

    c_list = [zeros(1, N_ens) for i in 1:nthreads]
    cp_list = [zeros(1, N_ens) for i in 1:nthreads]
    m_list = [zeros(output_dim, N_ens, n_test) for i in 1:nthreads]
    moc_list = [zeros(output_dim, output_dim, n_test) for i in 1:nthreads]
    buffer_list = [zeros(n_test, output_dim, n_features) for i in 1:nthreads]
    moc_tmp_list = [zeros(output_dim, output_dim, n_test) for i in 1:nthreads]
    mtmp_list = [zeros(output_dim, n_test) for i in 1:nthreads]
    rng_seed = randperm(rng, 10^5)[1] # dumb way to get a random integer in 1:10^5
    rng_list = [Random.MersenneTwister(rng_seed + i) for i in 1:nthreads]
    Threads.@threads for i in ProgressBar(1:N_ens)
        tid = Threads.threadid()
        rngtmp = rng_list[tid]
        mtmp = mtmp_list[tid]
        moc_tmp = moc_tmp_list[tid]
        buffer = buffer_list[tid]
        l = lmat[:, i]
        for j in collect(1:repeats)

            c, cplxty = calculate_mean_cov_and_coeffs(
                rfi,
                rngtmp,
                l,
                regularization,
                n_features,
                n_train,
                n_test,
                batch_sizes,
                io_pairs,
                decomp_type,
                mtmp,
                moc_tmp,
                buffer,
                multithread_type,
            )

            # m output_dim x n_test
            # v output_dim x output_dim x n_test
            # c n_features
            m_list[tid][:, i, :] += mtmp ./ repeats
            @. moc_list[tid] += moc_tmp ./ (repeats * N_ens)
            c_list[tid][1, i] += sqrt(sum(c .^ 2)) / repeats
            cp_list[tid][1, i] += cplxty / repeats

        end
    end

    # put back together after threading
    mean_of_covs = sum(moc_list)
    means = sum(m_list)
    coeffl2norm = sum(c_list)
    complexity = sum(cp_list)
    means = permutedims(means, (3, 2, 1))
    mean_of_covs = permutedims(mean_of_covs, (3, 1, 2))
    blockcovmat = zeros(n_test * output_dim, n_test * output_dim)
    blockmeans = zeros(n_test * output_dim, N_ens)
    for i in 1:n_test
        id = ((i - 1) * output_dim + 1):(i * output_dim)
        blockcovmat[id, id] = mean_of_covs[i, :, :]
        blockmeans[id, :] = permutedims(means[i, :, :], (2, 1))
    end

    if !isposdef(blockcovmat)
        println("blockcovmat not posdef")
        blockcovmat = posdef_correct(blockcovmat)
    end

    return vcat(blockmeans, coeffl2norm, complexity), blockcovmat

end

include("ScalarRandomFeature.jl")
include("VectorRandomFeature.jl")
