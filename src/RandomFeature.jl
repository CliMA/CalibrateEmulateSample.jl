
using RandomFeatures
const RF = RandomFeatures
using EnsembleKalmanProcesses
const EKP = EnsembleKalmanProcesses
using ..ParameterDistributions
using StableRNGs
using Random


export ScalarRandomFeatureInterface
export
    get_rfms,
    get_fitted_features,
    get_hyper_prior,
    get_batch_sizes,
    get_n_features,
    get_rng


struct ScalarRandomFeatureInterface <: MachineLearningTool
    "vector of `RandomFeatureMethod`s, contains the feature structure, batch-sizes and regularization"
    rfms::Vector{RF.Methods.RandomFeatureMethod}
    "vector of `Fit`s, containing the matrix decomposition and coefficients of RF when fitted to data"
    fitted_features::Vector{RF.Methods.Fit}
    "hyperparameter priors"
    hyper_prior::ParameterDistribution
    "batch sizes"
    batch_sizes::Union{Dict,Nothing}
    "n_features"
    n_features::Union{Int,Nothing}
    "rng"
    rng::AbstractRNG
end

get_rfms(srfi::ScalarRandomFeatureInterface) = srfi.rfms
get_fitted_features(srfi::ScalarRandomFeatureInterface) = srfi.fitted_features
get_hyper_prior(srfi::ScalarRandomFeatureInterface) = srfi.hyper_prior
get_batch_sizes(srfi::ScalarRandomFeatureInterface) = srfi.batch_sizes
get_n_features(srfi::ScalarRandomFeatureInterface) = srfi.n_features
get_rng(srfi::ScalarRandomFeatureInterface) = srfi.rng

function ScalarRandomFeatureInterface(
    n_features::Int,
    hyper_prior::ParameterDistribution;
    batch_sizes::Union{Dict,Nothing}=nothing,
    rng = Random.GLOBAL_RNG)
   # Initialize vector for GP models
    rfms = Vector{Union{RF.Methods.RandomFeatureMethod, Nothing}}(undef, 0) 
    fitted_features = Vector{Union{RF.Methods.Fit, Nothing}}(undef, 0)
    return ScalarRandomFeatureInterface(
        rfms,
        fitted_features,
        hyper_prior,
        batch_sizes,
        n_features,
        rng
    )
end


function RFM_from_hyperparameters(
    rng::AbstractRNG,
    l::Union{Real, AbstractVecOrMat},
    regularizer::Real,
    n_features::Int,
    batch_sizes::Union{Dict,Nothing},
    input_dim::Int,
)

    μ_c = 0.0
    if isa(l, Real)
        σ_c = fill(l, input_dim)
    elseif isa(l, AbstractVector)
        if length(l) == 1
            σ_c = fill(l[1], input_dim)
        else
            σ_c = l
        end
    else
        isa(l, AbstractMatrix)
        σ_c = l[:, 1]
    end

    pd = ParameterDistribution(
        Dict(
            "distribution" => VectorOfParameterized(map(sd -> Normal(μ_c, sd), σ_c)),
            "constraint" => repeat([no_constraint()], input_dim),
            "name" => "xi",
        ),
    )
    feature_sampler = RF.Samplers.FeatureSampler(pd, rng = rng)
    # Learn hyperparameters for different feature types

    sf = RF.Features.ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = Dict("sigma" => 1))
    if isnothing(batch_sizes)
        return RF.Methods.RandomFeatureMethod(sf, regularization = regularizer)
    else
        return RF.Methods.RandomFeatureMethod(sf, regularization = regularizer, batch_sizes = batch_sizes)
    end
end


function calculate_mean_cov_and_coeffs(
    rng::AbstractRNG,
    l::Union{Real, AbstractVecOrMat},
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Union{Dict,Nothing},
    io_pairs::PairedDataContainer,
)

    regularizer = noise_sd^2
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    # split data into train/test randomly
    itrain = get_inputs(io_pairs)[:, 1:n_train]
    otrain = get_outputs(io_pairs)[:, 1:n_train]
    io_train_cost = PairedDataContainer(itrain, otrain)
    itest = get_inputs(io_pairs)[:, (n_train + 1):end]
    otest = get_outputs(io_pairs)[:, (n_train + 1):end]
    input_dim = size(itrain, 1)

    # build and fit the RF
    rfm = RFM_from_hyperparameters(rng, l, regularizer, n_features, batch_sizes, input_dim)
    fitted_features = fit(rfm, io_train_cost, decomposition_type = "svd")

    test_batch_size = RF.Methods.get_batch_size(rfm, "test")
    batch_inputs = RF.Utilities.batch_generator(itest, test_batch_size, dims = 2) # input_dim x batch_size

    #we want to calc 1/var(y-mean)^2 + lambda/m * coeffs^2 in the end
    pred_mean, pred_cov = RF.Methods.predict(rfm, fitted_features, DataContainer(itest))
    scaled_coeffs = sqrt(1 / n_features) * RF.Methods.get_coeffs(fitted_features)
    return pred_mean, pred_cov, scaled_coeffs

end


function estimate_mean_cov_and_coeffnorm_covariance(
    rng::AbstractRNG,
    l::Union{Real, AbstractVecOrMat},
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Union{Dict,Nothing},
    io_pairs::PairedDataContainer,
    n_samples::Int;
    repeats::Int = 1,
)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train
    means = zeros(n_test, n_samples)
    covs = zeros(n_test, n_samples)
    coeffl2norm = zeros(1, n_samples)
    for i in 1:n_samples
        for j in 1:repeats
            m, v, c = calculate_mean_cov_and_coeffs(rng, l, noise_sd, n_features, batch_sizes, io_pairs)
            means[:, i] += m' / repeats
            covs[:, i] += v' / repeats
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
        end
    end
    #    println("take covariances")
    Γ = cov(vcat(means, covs, coeffl2norm), dims = 2)
    approx_σ2 = Diagonal(mean(covs, dims = 2)[:, 1]) # approx of \sigma^2I +rf var

    return Γ, approx_σ2

end

function calculate_ensemble_mean_cov_and_coeffnorm(
    rng::AbstractRNG,
    lvecormat::AbstractVecOrMat,
    noise_sd::Real,
    n_features::Int,
    batch_sizes::Union{Dict,Nothing},
    io_pairs::PairedDataContainer;
    repeats::Int = 1,
)
    if isa(lvecormat, AbstractVector)
        lmat = reshape(lvecormat, 1, :)
    else
        lmat = lvecormat
    end
    N_ens = size(lmat, 2)
    n_train = Int(floor(0.8 * size(get_inputs(io_pairs), 2))) # 80:20 train test
    n_test = size(get_inputs(io_pairs), 2) - n_train

    means = zeros(n_test, N_ens)
    covs = zeros(n_test, N_ens)
    coeffl2norm = zeros(1, N_ens)
    for i in collect(1:N_ens)
        for j in collect(1:repeats)
            l = lmat[:, i]
            m, v, c = calculate_mean_cov_and_coeffs(rng, l, noise_sd, n_features, batch_sizes, io_pairs)
            means[:, i] += m' / repeats
            covs[:, i] += v' / repeats
            coeffl2norm[1, i] += sqrt(sum(c .^ 2)) / repeats
        end
    end
    return vcat(means, covs, coeffl2norm), Diagonal(mean(covs, dims = 2)[:, 1]) # approx of \sigma^2I 


end


function build_models!(
    srfi::ScalarRandomFeatureInterface,
    input_output_pairs::PairedDataContainer{FT},
) where {FT <: AbstractFloat}
    # get inputs and outputs 
    input_values = get_inputs(input_output_pairs)
    output_values = get_outputs(input_output_pairs)
    n_rfms, n_data = size(output_values)
    noise_sd = 1.0
    
    input_dim = size(input_values,1)
    
    rfms = get_rfms(srfi)
    fitted_features = get_fitted_features(srfi)
    hyper_prior = get_hyper_prior(srfi)
    batch_sizes = get_batch_sizes(srfi)
    rng = get_rng(srfi)

    # Optimize features with EKP for each output dim
    # [1.] Split data into test/train 80/20
    n_train = Int(floor(0.8 * n_data))
    n_test = n_data - n_train
    n_features_opt = Int(floor(1.2 * n_test)) #we take a specified number of features for optimization.

    for i = 1:n_rfms
       
        io_pairs_i = PairedDataContainer(
            input_values,
            reshape(output_values[i,:],1,size(output_values,2)),
        )
        
        # [2.] Estimate covariance at mean value
        μ_hp = transform_unconstrained_to_constrained(hyper_prior,mean(hyper_prior))
        cov_samples = 2 * Int(floor(n_test))
        internal_Γ, approx_σ2 = estimate_mean_cov_and_coeffnorm_covariance(
            rng,
            μ_hp,
            1.0,
            n_features_opt,
            batch_sizes,
            io_pairs_i,
            cov_samples,
        )
        
        Γ = internal_Γ
        Γ[1:n_test, 1:n_test] += approx_σ2
        Γ[(n_test + 1):(2 * n_test), (n_test + 1):(2 * n_test)] += I
        Γ[(2 * n_test + 1):end, (2 * n_test + 1):end] += I
        
        # [3.] set up EKP optimization
        N_ens = 20 * input_dim
        N_iter = 30
        
        initial_params = construct_initial_ensemble(rng, hyper_prior, N_ens)
        params_init = transform_unconstrained_to_constrained(hyper_prior, initial_params)[1, :]
        data = vcat(get_outputs(io_pairs_i)[(n_train + 1):end], noise_sd^2 * ones(n_test), 0.0)
        ekiobj = EKP.EnsembleKalmanProcess(initial_params, data, Γ, Inversion(), rng = rng)
        err = zeros(N_iter)

        # [4.] optimize with EKP
        for i = 1:N_iter
            
            #get parameters:
            lvec = transform_unconstrained_to_constrained(hyper_prior, get_u_final(ekiobj))
            g_ens, _ = calculate_ensemble_mean_cov_and_coeffnorm(
                rng,
                lvec,
                1.0,
                n_features_opt,
                batch_sizes,
                io_pairs_i,
            )
            EKP.update_ensemble!(ekiobj, g_ens)
            err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
            
        end
        
        # [5.] extract optimal hyperparameters
        l_optimal =  mean(transform_unconstrained_to_constrained(hyper_prior, get_u_final(ekiobj)), dims = 2)
        println("RFM $i, learnt hyperparameters: $(l_optimal)")
        # Now, fit new RF model with the optimized hyperparameters
        n_features = get_n_features(srfi)
        
        μ_c = 0.0
        if size(l_optimal, 1) == 1
            σ_c = repeat([l_optimal[1, 1]], input_dim)
        else
            σ_c = l_optimal[:, 1]
        end
        pd = ParameterDistribution(
            Dict(
                "distribution" => VectorOfParameterized(map(sd -> Normal(μ_c, sd), σ_c)),
                "constraint" => repeat([no_constraint()], input_dim),
                "name" => "xi",
            ),
        )
        feature_sampler = RF.Samplers.FeatureSampler(pd, rng = copy(rng))
        sigma_fixed = Dict("sigma" => 1.0)
        sff = RF.Features.ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = sigma_fixed) #build scalar fourier features
        if isnothing(batch_sizes)#setup random feature method
            rfm_i = RF.Methods.RandomFeatureMethod(sff, regularization = 1.0) 
        else
            rfm_i = RF.Methods.RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = 1.0)
        end
        fitted_features_i = RF.Methods.fit(rfm_i, io_pairs_i, decomposition_type = "svd") #fit features
        
        push!(rfms, rfm_i)
        push!(fitted_features, fitted_features_i)
    end
    
end

function optimize_hyperparameters!(srfi::ScalarRandomFeatureInterface, args...; kwargs...)
    println("Random Features already trained. continuing...")
end


function predict(
    srfi::ScalarRandomFeatureInterface,
    new_inputs::AbstractMatrix{FT},
) where {FT <: AbstractFloat}
    M = length(get_rfms(srfi))
    N_samples = size(new_inputs, 2)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(M, N_samples)
    σ2 = zeros(M, N_samples)
    for i in 1:M
        μ[i, :], σ2[i, :] = RF.Methods.predict(
            get_rfms(srfi)[i],
            get_fitted_features(srfi)[i],
            DataContainer(new_inputs),
        )
    end

    # add the noise contribution from the regularization
   σ2[:,:] = σ2[:,:] .+ 1.0
    
    return μ, σ2
end
