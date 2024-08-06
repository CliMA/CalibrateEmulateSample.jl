
using RandomFeatures
const RF = RandomFeatures
using EnsembleKalmanProcesses
const EKP = EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Localizers
using ..ParameterDistributions
using ..Utilities
using StableRNGs
using Random

export calculate_n_hyperparameters, hyperparameters_from_flat, build_default_prior, cov_structure_from_string
export CovarianceStructureType, OneDimFactor, DiagonalFactor, CholeskyFactor, LowRankFactor, HierarchicalLowRankFactor
export SeparableKernel, NonseparableKernel
export get_input_cov_structure, get_output_cov_structure, get_cov_structure
export get_eps
export rank
export shrinkage_cov, nice_cov


abstract type RandomFeatureInterface <: MachineLearningTool end

# Different types of threading
abstract type MultithreadType end
struct TullioThreading <: MultithreadType end
struct EnsembleThreading <: MultithreadType end


# Different types of covariance representation in the prior

abstract type CovarianceStructureType end

"""
$(DocStringExtensions.TYPEDEF)

covariance structure for a one-dimensional space
"""
struct OneDimFactor <: CovarianceStructureType end

"""
$(DocStringExtensions.TYPEDEF)

builds a diagonal covariance structure
"""
struct DiagonalFactor{FT <: AbstractFloat} <: CovarianceStructureType
    "nugget term"
    eps::FT
end
DiagonalFactor() = DiagonalFactor(Float64(1.0))
get_eps(df::DiagonalFactor) = df.eps

"""
$(DocStringExtensions.TYPEDEF)

builds a general positive-definite covariance structure
"""
struct CholeskyFactor{FT <: AbstractFloat} <: CovarianceStructureType
    "nugget term"
    eps::FT
end
CholeskyFactor() = CholeskyFactor(Float64(1.0))
get_eps(cf::CholeskyFactor) = cf.eps

"""
$(DocStringExtensions.TYPEDEF)

builds a covariance structure that deviates from the identity with a low-rank perturbation. This perturbation is diagonalized in the low-rank space
"""
struct LowRankFactor{FT <: AbstractFloat} <: CovarianceStructureType
    "rank of perturbation"
    rank::Int
    "nugget term"
    eps::FT
end
LowRankFactor(r::Int) = LowRankFactor(r, Float64(1.0))
get_eps(lrf::LowRankFactor) = lrf.eps


"""
$(DocStringExtensions.TYPEDEF)

builds a covariance structure that deviates from the identity with a more general low-rank perturbation
"""
struct HierarchicalLowRankFactor{FT <: AbstractFloat} <: CovarianceStructureType
    "rank of perturbation"
    rank::Int
    "nugget term"
    eps::FT
end

HierarchicalLowRankFactor(r::Int) = HierarchicalLowRankFactor(r, Float64(1.0))
get_eps(hlrf::HierarchicalLowRankFactor) = hlrf.eps

import LinearAlgebra: rank
rank(lrf::LowRankFactor) = lrf.rank
rank(hlrf::HierarchicalLowRankFactor) = hlrf.rank

# To add a new cov type `T`  one must define these 3 methods:
# hyperparameters_from_flat(x::V, d::Int, t::T) where {V <: AbstractVector}
# calculate_n_hyperparameters(d::Int, t::T)
# build_default_prior(name::SS, d::Int, t::T) where {SS <: AbstractString}
# and add a string id to cov_structure_from_string

"""
$(DocStringExtensions.TYPEDSIGNATURES)

creates some simple covariance structures from strings: ["onedim", "diagonal", "cholesky", "lowrank", hierlowrank"]. See the covariance Structures for more details.
"""
function cov_structure_from_string(s::S, d::Int) where {S <: AbstractString}
    if s == "onedim"
        return OneDimFactor()
    elseif s == "diagonal"
        return DiagonalFactor()
    elseif s == "cholesky"
        return CholeskyFactor()
    elseif s == "lowrank"
        return LowRankFactor(Int(ceil(sqrt(d))))
    elseif s == "hierlowrank"
        return HierarchicalLowRankFactor(Int(ceil(sqrt(d))))
    else
        throw(
            ArgumentError(
                "Recieved unknown `input_cov_structure` keyword $(s), \n please choose from [\"onedim\",\"diagonal\",\"cholesky\", \"lowrank\"] or provide a CovarianceStructureType",
            ),
        )
    end
end
cov_structure_from_string(cst::CST, args...; kwargs...) where {CST <: CovarianceStructureType} = cst


# Different types of kernel 
abstract type KernelStructureType end

"""
$(DocStringExtensions.TYPEDEF)

Builds a separable kernel, i.e. one that accounts for input and output covariance structure separately
"""
struct SeparableKernel{CST1 <: CovarianceStructureType, CST2 <: CovarianceStructureType} <: KernelStructureType
    input_cov_structure::CST1
    output_cov_structure::CST2
end
get_input_cov_structure(kernel_structure::SeparableKernel) = kernel_structure.input_cov_structure
get_output_cov_structure(kernel_structure::SeparableKernel) = kernel_structure.output_cov_structure


"""
$(DocStringExtensions.TYPEDEF)

Builds a nonseparable kernel, i.e. one that accounts for a joint input and output covariance structure
"""
struct NonseparableKernel{CST <: CovarianceStructureType} <: KernelStructureType
    cov_structure::CST
end
get_cov_structure(kernel_structure::NonseparableKernel) = kernel_structure.cov_structure


# calculate_n_hyperparameters

"""
$(DocStringExtensions.TYPEDSIGNATURES)

calculates the number of hyperparameters generated by the choice of covariance structure
"""
calculate_n_hyperparameters(d::Int, odf::OneDimFactor) = 0
calculate_n_hyperparameters(d::Int, df::DiagonalFactor) = d
calculate_n_hyperparameters(d::Int, cf::CholeskyFactor) = Int(d * (d + 1) / 2)
calculate_n_hyperparameters(d::Int, lrf::LowRankFactor) = Int(rank(lrf) + d * rank(lrf))
calculate_n_hyperparameters(d::Int, hlrf::HierarchicalLowRankFactor) =
    Int(rank(hlrf) * (rank(hlrf) + 1) / 2 + d * rank(hlrf))

# build from flat 


"""
$(DocStringExtensions.TYPEDSIGNATURES)

reshapes a list of hyperparameters into a covariance matrix based on the selected structure
"""
hyperparameters_from_flat(x::V, odf::OneDimFactor) where {V <: AbstractVector} = nothing

function hyperparameters_from_flat(x::V, df::DiagonalFactor) where {V <: AbstractVector}
    return convert(Matrix, Diagonal(x) + get_eps(df) * I)
end

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

function hyperparameters_from_flat(x::V, cf::CholeskyFactor) where {V <: AbstractVector}
    chol = flat_to_chol(x)
    return chol * permutedims(chol, (2, 1)) + get_eps(cf) * I

end

function hyperparameters_from_flat(x::V, lrf::LowRankFactor) where {V <: AbstractVector}

    r = rank(lrf)
    d = Int(length(x) / r - 1) # l = r + d*r

    D = Diagonal(x[1:r]) # D acts like sing.vals.^2
    U = reshape(x[(r + 1):end], d, r)
    qrU = qr(U) # qrU.Q is semi-orthogonal in first r columns, i.e. Q^T.Q = I
    return qrU.Q[:, 1:r] * D * qrU.Q[:, 1:r]' + get_eps(lrf) * I

end

function hyperparameters_from_flat(x::V, hlrf::HierarchicalLowRankFactor) where {V <: AbstractVector}
    r = rank(hlrf)
    d = Int(length(x) / r - (r + 1) / 2) # l = r^2 + d*r

    # Ambikasaran, O'Neil, Singh 2016
    L = flat_to_chol(x[1:Int(r * (r + 1) / 2)])
    U = reshape(x[Int(r * (r + 1) / 2 + 1):end], d, r)
    K = L * permutedims(L, (2, 1)) + get_eps(hlrf) * I

    qrU = qr(U)
    svdT = svd(I + qrU.R * K * qrU.R') # square so T = Q S Q^t 
    M = svdT.V * Diagonal(sqrt.(svdT.S)) # M M^t = Q sqrt(S) (Q sqrt(S))^t = Q sqrt(S) sqrt(S) Q^t = T
    X = M - I
    update_factor = I + qrU.Q * X * qrU.Q'

    return update_factor * update_factor'
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

builds a prior distribution for the kernel hyperparameters to initialize optimization.
"""
build_default_prior(name::SS, n_hp::Int, odf::OneDimFactor) where {SS <: AbstractString} = nothing

function build_default_prior(name::SS, n_hp::Int, df::DiagonalFactor) where {SS <: AbstractString}
    return ParameterDistribution(
        Dict(
            "name" => "$(name)_diagonal",
            "distribution" => VectorOfParameterized(repeat([Normal(0, 3)], n_hp)),
            "constraint" => repeat([bounded_below(0.0)], n_hp),
        ),
    )
end

function build_default_prior(name::SS, n_hp::Int, df::CholeskyFactor) where {SS <: AbstractString}
    return constrained_gaussian("$(name)_cholesky", 0.0, 5.0, -Inf, Inf, repeats = n_hp)
end

function build_default_prior(name::SS, n_hp::Int, lrf::LowRankFactor) where {SS <: AbstractString}
    r = rank(lrf)
    d = Int(n_hp / r - 1)
    D = ParameterDistribution(
        Dict(
            "name" => "$(name)_lowrank_diagonal",
            "distribution" => VectorOfParameterized(repeat([Normal(0, 3)], r)),
            "constraint" => repeat([bounded_below(0.0)], r),
        ),
    )
    U = constrained_gaussian("$(name)_lowrank_U", 0.0, 100.0, -Inf, Inf, repeats = Int(d * r))
    return combine_distributions([D, U])
end

function build_default_prior(name::SS, n_hp::Int, hlrf::HierarchicalLowRankFactor) where {SS <: AbstractString}
    r = rank(hlrf)
    d = Int(n_hp / r - (r + 1) / 2)
    L = constrained_gaussian("$(name)_lowrank_Kchol", 0.0, 5.0, -Inf, Inf, repeats = Int(r * (r + 1) / 2))
    U = constrained_gaussian("$(name)_lowrank_U", 0.0, 100.0, -Inf, Inf, repeats = Int(d * r))
    return combine_distributions([L, U])
end

# combining input and output spaces:
function calculate_n_hyperparameters(
    input_dim::Int,
    output_dim::Int,
    kernel_structure::SK,
) where {SK <: SeparableKernel}

    input_cov_structure = get_input_cov_structure(kernel_structure)
    n_hp_in = calculate_n_hyperparameters(input_dim, input_cov_structure)
    output_cov_structure = get_output_cov_structure(kernel_structure)
    n_hp_out = calculate_n_hyperparameters(output_dim, output_cov_structure)
    n_scalings = 1

    # if both in and output dim are "OneDimFactors" we still learn 1 hp
    if n_hp_in + n_hp_out == 0
        return 1 + n_scalings
    end

    return n_hp_in + n_hp_out + n_scalings
end

function calculate_n_hyperparameters(input_dim::Int, kernel_structure::KST) where {KST <: KernelStructureType}
    output_dim = 1
    return calculate_n_hyperparameters(input_dim, output_dim, kernel_structure)
end

function calculate_n_hyperparameters(
    input_dim::Int,
    output_dim::Int,
    kernel_structure::NK,
) where {NK <: NonseparableKernel}
    cov_structure = get_cov_structure(kernel_structure)
    n_hp = calculate_n_hyperparameters(input_dim * output_dim, cov_structure)
    n_scalings = 1
    # if both in and output dim are "OneDimFactors" we still learn 1 hp
    if n_hp == 0
        return 1 + n_scalings
    end

    return n_hp + n_scalings
end




function build_default_prior(input_dim::Int, output_dim::Int, kernel_structure::SK) where {SK <: SeparableKernel}
    input_cov_structure = get_input_cov_structure(kernel_structure)
    output_cov_structure = get_output_cov_structure(kernel_structure)

    n_hp_in = calculate_n_hyperparameters(input_dim, input_cov_structure)
    input_prior = build_default_prior("input", n_hp_in, input_cov_structure)
    n_hp_out = calculate_n_hyperparameters(output_dim, output_cov_structure)
    output_prior = build_default_prior("output", n_hp_out, output_cov_structure)

    scaling_kern = constrained_gaussian("sigma", 1, 5, 0, Inf)

    # We only use OneDimFactor values, default to input if both in and output dimension are one dimensional.Otherwise they are ignored
    if isnothing(input_prior) && isnothing(output_prior)
        onedim_prior = constrained_gaussian("onedim", 1.0, 1.0, 0.0, Inf)
        return combine_distributions([onedim_prior, scaling_kern])
    elseif isnothing(input_prior)
        return combine_distributions([output_prior, scaling_kern])
    elseif isnothing(output_prior)
        return combine_distributions([input_prior, scaling_kern])
    else
        return combine_distributions([input_prior, output_prior, scaling_kern])
    end
end

function build_default_prior(input_dim::Int, kernel_structure::KST) where {KST <: KernelStructureType}
    output_dim = 1
    return build_default_prior(input_dim, output_dim, kernel_structure)
end

function build_default_prior(input_dim::Int, output_dim::Int, kernel_structure::NK) where {NK <: NonseparableKernel}
    cov_structure = get_cov_structure(kernel_structure)
    n_hp = calculate_n_hyperparameters(input_dim * output_dim, cov_structure)
    if n_hp == 0
        pd_kern = constrained_gaussian("onedim", 1.0, 1.0, 0.0, Inf)
    else
        pd_kern = build_default_prior("full", n_hp, cov_structure)
    end

    scaling_kern = constrained_gaussian("sigma", 1, 5, 0, Inf)
    return combine_distributions([pd_kern, scaling_kern])
end

function hyperparameters_from_flat(
    x::VV,
    input_dim::Int,
    output_dim::Int,
    kernel_structure::SK,
) where {VV <: AbstractVector, SK <: SeparableKernel}
    input_cov_structure = get_input_cov_structure(kernel_structure)
    output_cov_structure = get_output_cov_structure(kernel_structure)

    n_hp_in = calculate_n_hyperparameters(input_dim, input_cov_structure)
    U = hyperparameters_from_flat(x[1:n_hp_in], input_cov_structure)
    V = hyperparameters_from_flat(x[(n_hp_in + 1):end], output_cov_structure)

    if isnothing(U) && isnothing(V) #i.e. both are from OneDimFactor cov structures
        return x[1] * ones(1, 1), V
    elseif isnothing(U)
        return U, 0.5 * (V + permutedims(V, (2, 1)))
    elseif isnothing(V)
        return 0.5 * (U + permutedims(U, (2, 1))), V
    else
        # symmetrize (sometimes there are numerical artifacts)
        U = 0.5 * (U + permutedims(U, (2, 1)))
        V = 0.5 * (V + permutedims(V, (2, 1)))
        return U, V
    end
end

function hyperparameters_from_flat(
    x::VV,
    input_dim::Int,
    kernel_structure::SK,
) where {VV <: AbstractVector, SK <: SeparableKernel}
    output_dim = 1
    U, V = hyperparameters_from_flat(x, input_dim, output_dim, kernel_structure)

    return U
end

function hyperparameters_from_flat(
    x::VV,
    input_dim::Int,
    output_dim::Int,
    kernel_structure::NK,
) where {VV <: AbstractVector, NK <: NonseparableKernel}
    cov_structure = get_cov_structure(kernel_structure)
    C = hyperparameters_from_flat(x, cov_structure)
    if isnothing(C)
        return x[1] * ones(1, 1)
    else
        return 0.5 * (C + permutedims(C, (2, 1)))
    end

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
    n_test = size(itest, 2)
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

    ## TODO - the theory states that the following should be set:
    #    scaled_coeffs = sqrt(1 / (n_features)) * RF.Methods.get_coeffs(fitted_features)
    scaled_coeffs = 1e-3 * rand(n_features)#overwrite with noise...
    # However the convergence is much improved with setting this to zero:
    #scaled_coeffs = 0    
    if decomp_type == "cholesky"
        chol_fac = RF.Methods.get_decomposition(RF.Methods.get_feature_factors(fitted_features)).L

        complexity = 2 * sum(log(chol_fac[i, i]) for i in 1:size(chol_fac, 1))
    else
        svd_singval = RF.Methods.get_decomposition(RF.Methods.get_feature_factors(fitted_features)).S
        complexity = sum(log, svd_singval) # note this is log(abs(det))
    end
    complexity = sqrt(complexity)
    return scaled_coeffs, complexity

end



"""
$(DocStringExtensions.TYPEDSIGNATURES)

Calculate the empirical covariance, additionally applying a shrinkage operator (here the Ledoit Wolf 2004 shrinkage operation). Known to have better stability properties than Monte-Carlo for low sample sizes
"""
function shrinkage_cov(sample_mat::AA) where {AA <: AbstractMatrix}
    n_out, n_sample = size(sample_mat)

    # de-mean (as we will use the samples directly for calculation of β)
    sample_mat_zeromean = sample_mat .- mean(sample_mat, dims = 2)
    # Ledoit Wolf shrinkage to I

    # get sample covariance
    Γ = cov(sample_mat_zeromean, dims = 2)
    # estimate opt shrinkage
    μ_shrink = 1 / n_out * tr(Γ)
    δ_shrink = norm(Γ - μ_shrink * I)^2 / n_out # (scaled) frob norm of Γ_m
    #once de-meaning, we need to correct the sample covariance with an n_sample -> n_sample-1
    β_shrink = sum([norm(c * c' - -Γ)^2 / n_out for c in eachcol(sample_mat_zeromean)]) / (n_sample - 1)^2

    γ_shrink = min(β_shrink / δ_shrink, 1) # clipping is typically rare
    #  γμI + (1-γ)Γ
    Γ .*= (1 - γ_shrink)
    for i in 1:n_out
        Γ[i, i] += γ_shrink * μ_shrink
    end

    @info "Shrinkage scale: $(γ_shrink), (0 = none, 1 = revert to scaled Identity)\n shrinkage covariance condition number: $(cond(Γ))"
    return Γ
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Calculate the empirical covariance, additionally applying the Noise Informed Covariance Estimator (NICE) Vishnay et al. 2024.
"""
function nice_cov(sample_mat::AA, n_samples = 400, δ::FT = 1.0) where {AA <: AbstractMatrix, FT <: Real}

    n_sample_cov = size(sample_mat, 2)
    Γ = cov(sample_mat, dims = 2)

    bd_tol = 1e8 * eps()

    v = sqrt.(diag(Γ))
    V = Diagonal(v) #stds
    V_inv = inv(V)
    corr = clamp.(V_inv * Γ * V_inv, -1 + bd_tol, 1 - bd_tol) # full corr

    # parameter sweep over the exponents
    max_exponent = 2 * 5 # must be even
    interp_steps = 100
    # use find the variability in the corr coeff matrix entries
    std_corrs = approximate_corr_std.(corr, n_sample_cov, n_samples) # Found in EKP.Localizers !! slowest part of code -> could speed up by precomputing an interpolation of [-1,1]

    std_tol = sqrt(sum(std_corrs .^ 2))
    α_min_exceeded = [max_exponent]
    for α in 2:2:max_exponent # even exponents give a PSD
        corr_psd = corr .^ (α + 1) # abs not needed as α even
        # find the first exponent that exceeds the noise tolerance in norm
        if norm(corr_psd - corr) > δ * std_tol
            α_min_exceeded[1] = α
            break
        end
    end
    corr_psd = corr .^ α_min_exceeded[1]
    corr_psd_prev = corr .^ (α_min_exceeded[1] - 2) # previous PSD correction 

    for α in LinRange(1.0, 0.0, interp_steps)
        corr_interp = ((1 - α) * (corr_psd_prev) + α * corr_psd) .* corr
        if norm(corr_interp - corr) < δ * std_tol
            corr[:, :] = corr_interp #update the correlation matrix block
            break
        end
    end
    out = posdef_correct(V * corr * V) # rebuild the cov matrix
    @info "NICE-adjusted covariance condition number: $(cond(out))"
    return out

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
    cov_correction = "shrinkage",
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

    sample_mat = vcat(blockmeans, coeffl2norm, complexity)

    if cov_correction == "shrinkage"
        Γ = shrinkage_cov(sample_mat)
    elseif cov_correction == "nice"
        Γ = nice_cov(sample_mat)
    else
        Γ = cov(sample_mat, dims = 2)
    end

    if !isposdef(approx_σ2)
        println("approx_σ2 not posdef")
        approx_σ2 = posdef_correct(approx_σ2)
    end

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
    cov_correction = "shrinkage",
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


    sample_mat = vcat(blockmeans, coeffl2norm, complexity)

    if cov_correction == "shrinkage"
        Γ = shrinkage_cov(sample_mat)
    elseif cov_correction == "nice"
        Γ = nice_cov(sample_mat)
    else
        Γ = cov(sample_mat, dims = 2)
    end

    if !isposdef(approx_σ2)
        println("approx_σ2 not posdef")
        approx_σ2 = posdef_correct(approx_σ2)
    end

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
