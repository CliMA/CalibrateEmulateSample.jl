export ScalarRandomFeatureInterface
# getters already exported in VRFI

"""
$(DocStringExtensions.TYPEDEF)

Structure holding the Scalar Random Feature models. 

# Fields
$(DocStringExtensions.TYPEDFIELDS)

"""
struct ScalarRandomFeatureInterface{S <: AbstractString, RNG <: AbstractRNG, KST <: KernelStructureType} <:
       RandomFeatureInterface
    "vector of `RandomFeatureMethod`s, contains the feature structure, batch-sizes and regularization"
    rfms::Vector{RF.Methods.RandomFeatureMethod}
    "vector of `Fit`s, containing the matrix decomposition and coefficients of RF when fitted to data"
    fitted_features::Vector{RF.Methods.Fit}
    "batch sizes"
    batch_sizes::Union{Dict{S, Int}, Nothing}
    "n_features"
    n_features::Union{Int, Nothing}
    "input dimension"
    input_dim::Int
    "choice of random number generator"
    rng::RNG
    "regularization"
    regularization::Vector{Union{Matrix, UniformScaling, Diagonal}}
    "Kernel structure type (e.g. Separable or Nonseparable)"
    kernel_structure::KST
    "Random Feature decomposition, choose from \"svd\" or \"cholesky\" (default)"
    feature_decomposition::S
    "dictionary of options for hyperparameter optimizer"
    optimizer_options::Dict{S}
    "diagnostics from optimizer"
    optimizer::Vector
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the rfms field
"""
get_rfms(srfi::ScalarRandomFeatureInterface) = srfi.rfms

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the fitted_features field
"""
get_fitted_features(srfi::ScalarRandomFeatureInterface) = srfi.fitted_features

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets batch_sizes the field
"""
get_batch_sizes(srfi::ScalarRandomFeatureInterface) = srfi.batch_sizes

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the n_features field
"""
get_n_features(srfi::ScalarRandomFeatureInterface) = srfi.n_features

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the input_dim field
"""
get_input_dim(srfi::ScalarRandomFeatureInterface) = srfi.input_dim

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the rng field
"""
EKP.get_rng(srfi::ScalarRandomFeatureInterface) = srfi.rng

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the regularization field
"""
get_regularization(srfi::ScalarRandomFeatureInterface) = srfi.regularization


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the kernel_structure field
"""
get_kernel_structure(srfi::ScalarRandomFeatureInterface) = srfi.kernel_structure

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the feature_decomposition field
"""
get_feature_decomposition(srfi::ScalarRandomFeatureInterface) = srfi.feature_decomposition

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the optimizer_options field
"""
get_optimizer_options(srfi::ScalarRandomFeatureInterface) = srfi.optimizer_options

"""
$(DocStringExtensions.TYPEDSIGNATURES)

gets the optimizer field
"""
get_optimizer(srfi::ScalarRandomFeatureInterface) = srfi.optimizer


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructs a `ScalarRandomFeatureInterface <: MachineLearningTool` interface for the `RandomFeatures.jl` package for multi-input and single- (or decorrelated-)output emulators.
 - `n_features` - the number of random features
 - `input_dim` - the dimension of the input space
 - `kernel_structure` -  - a prescribed form of kernel structure
 - `batch_sizes = nothing` - Dictionary of batch sizes passed `RandomFeatures.jl` object (see definition there)
 - `rng = Random.GLOBAL_RNG` - random number generator 
 - `feature_decomposition = "cholesky"` - choice of how to store decompositions of random features, `cholesky` or `svd` available
 - `optimizer_options = nothing` - Dict of options to pass into EKI optimization of hyperparameters (defaults created in `ScalarRandomFeatureInterface` constructor):
     - "prior":  the prior for the hyperparameter optimization 
     - "prior_in_scale": use this to tune the input prior scale
     - "n_ensemble":  number of ensemble members
     - "n_iteration":  number of eki iterations
     - "cov_sample_multiplier": increase for more samples to estimate covariance matrix in optimization (default 10.0, minimum 0.0)  
     - "scheduler": Learning rate Scheduler (a.k.a. EKP timestepper) Default: DataMisfitController
     - "tikhonov":  tikhonov regularization parameter if >0
     - "inflation":  additive inflation ∈ [0,1] with 0 being no inflation
     - "train_fraction":  e.g. 0.8 (default)  means 80:20 train - test split
     - "n_features_opt":  fix the number of features for optimization (default `n_features`, as used for prediction)
     - "multithread": how to multithread. "ensemble" (default) threads across ensemble members "tullio" threads random feature matrix algebra
     - "accelerator": use EKP accelerators (default is no acceleration)
     - "verbose" => false, verbose optimizer statements
     - "cov_correction" => "shrinkage", type of conditioning to improve estimated covariance (Ledoit Wolfe 03), also "nice" for (Vishny, Morzfeld et al. 2024)
     - "n_cross_val_sets" => 2, train fraction creates (default 5) train-test data subsets, then use 'n_cross_val_sets' of these stacked in the loss function. If set to 0, train=test on the full data provided ignoring "train_fraction".
"""
function ScalarRandomFeatureInterface(
    n_features::Int,
    input_dim::Int;
    kernel_structure::Union{KST, Nothing} = nothing,
    batch_sizes::Union{Dict{S, Int}, Nothing} = nothing,
    rng::RNG = Random.GLOBAL_RNG,
    feature_decomposition::S = "cholesky",
    optimizer_options::Union{Dict{S}, Nothing} = nothing,
) where {S <: AbstractString, RNG <: AbstractRNG, KST <: KernelStructureType}
    # Initialize vector for GP models
    rfms = Vector{RF.Methods.RandomFeatureMethod}(undef, 0)
    fitted_features = Vector{RF.Methods.Fit}(undef, 0)
    regularization = Vector{Union{Matrix, UniformScaling, Nothing}}(undef, 0)

    if isnothing(kernel_structure)
        kernel_structure = SeparableKernel(cov_structure_from_string("lowrank", input_dim), OneDimFactor())
    end
    KSType = typeof(kernel_structure)

    prior = build_default_prior(input_dim, kernel_structure)

    # default optimizer settings
    optimizer_opts = Dict(
        "prior" => prior, #the hyperparameter_prior
        "n_ensemble" => min(10 * ndims(prior), 100), #number of ensemble
        "n_iteration" => 10, # number of eki iterations
        "scheduler" => EKP.DataMisfitController(terminate_at = 1000), # Adaptive timestepping,
        "cov_sample_multiplier" => 10.0, # multiplier for samples to estimate covariance in optimization scheme
        "inflation" => 1e-4, # additive inflation ∈ [0,1] with 0 being no inflation
        "train_fraction" => 0.8, # 80:20 train - test split
        "n_features_opt" => n_features, # number of features for the optimization 
        "multithread" => "ensemble", # instead of "tullio"
        "verbose" => false, # verbose optimizer statements
        "accelerator" => EKP.NesterovAccelerator(), # acceleration with momentum
        "localization" => EKP.Localizers.NoLocalization(), # localization / sample error correction for small ensembles
        "cov_correction" => "shrinkage", # type of conditioning to improve estimated covariance
        "n_cross_val_sets" => 2, # if >1 do cross validation, else if 0 do no data splitting and no training fraction
    )

    if !isnothing(optimizer_options)
        for key in keys(optimizer_options)
            optimizer_opts[key] = optimizer_options[key]
        end
    end

    opt_tmp = Dict()
    for key in keys(optimizer_opts)
        if key != "prior"
            opt_tmp[key] = optimizer_opts[key]
        end
    end
    @info("hyperparameter optimization with EKI configured with $opt_tmp")

    return ScalarRandomFeatureInterface{S, RNG, KSType}(
        rfms,
        fitted_features,
        batch_sizes,
        n_features,
        input_dim,
        rng,
        regularization,
        kernel_structure,
        feature_decomposition,
        optimizer_opts,
        [],
    )
end

function hyperparameter_distribution_from_flat(
    x::VV,
    input_dim::Int,
    kernel_structure::SK,
) where {VV <: AbstractVector, SK <: SeparableKernel}

    M = zeros(input_dim) #scalar output
    U = hyperparameters_from_flat(x, input_dim, kernel_structure)

    if !isposdef(U)
        println("U not posdef - correcting")
        U = posdef_correct(U)
    end

    dist = MvNormal(M, U)
    pd = ParameterDistribution(
        Dict(
            "distribution" => Parameterized(dist),
            "constraint" => repeat([no_constraint()], input_dim),
            "name" => "xi",
        ),
    )

    return pd
end

function hyperparameter_distribution_from_flat(
    x::VV,
    input_dim::Int,
    kernel_structure::NK,
) where {VV <: AbstractVector, NK <: NonseparableKernel}
    throw(
        ArgumentError(
            "Scalar Kernels must be of type `Separable( *** , OneDimFactor())`, received $(kernel_structure)",
        ),
    )
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the random feature method from hyperparameters. We use cosine activation functions and a MatrixVariateNormal(M,U,V) distribution (from `Distributions.jl`) with mean M=0, and input covariance U built using a `CovarianceStructureType`.
"""
function RFM_from_hyperparameters(
    srfi::ScalarRandomFeatureInterface,
    rng::RNG,
    l::ForVM,
    regularization::MorUSorD, # just a 1x1 matrix though
    n_features::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    input_dim::Int,
    multithread_type::MT;
) where {
    RNG <: AbstractRNG,
    ForVM <: Union{Real, AbstractVecOrMat},
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
    S <: AbstractString,
    MT <: MultithreadType,
}

    xi_hp = l[1:(end - 1)]
    sigma_hp = l[end]
    kernel_structure = get_kernel_structure(srfi)
    pd = hyperparameter_distribution_from_flat(xi_hp, input_dim, kernel_structure)

    feature_sampler = RF.Samplers.FeatureSampler(pd, rng = rng)
    # Learn hyperparameters for different feature types
    feature_parameters = Dict("sigma" => sigma_hp)
    sff = RF.Features.ScalarFourierFeature(n_features, feature_sampler, feature_parameters = feature_parameters)
    thread_opt = isa(multithread_type, TullioThreading) # if we want to multithread with tullio
    if isnothing(batch_sizes)
        return RF.Methods.RandomFeatureMethod(sff, regularization = regularization, tullio_threading = thread_opt)
    else
        return RF.Methods.RandomFeatureMethod(
            sff,
            regularization = regularization,
            batch_sizes = batch_sizes,
            tullio_threading = thread_opt,
        )
    end
end

#removes vector-only input arguments
RFM_from_hyperparameters(
    srfi::ScalarRandomFeatureInterface,
    rng::RNG,
    l::ForVM,
    regularization::MorUS, # just a 1x1 matrix though
    n_features::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    input_dim::Int,
    output_dim::Int,
    multithread_type::MT;
) where {
    RNG <: AbstractRNG,
    ForVM <: Union{Real, AbstractVecOrMat},
    MorUS <: Union{AbstractMatrix, UniformScaling},
    S <: AbstractString,
    MT <: MultithreadType,
} = RFM_from_hyperparameters(srfi, rng, l, regularization, n_features, batch_sizes, input_dim, multithread_type)

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the random feature method from hyperparameters. We use cosine activation functions and a Multivariate Normal distribution (from `Distributions.jl`) with mean M=0, and input covariance U built with the `CovarianceStructureType`.
"""
function build_models!(
    srfi::ScalarRandomFeatureInterface,
    input_output_pairs::PairedDataContainer{FT},
    input_structure_matrix,
    output_structure_matrix;
    kwargs...,
) where {FT <: AbstractFloat}

    # get inputs and outputs 
    input_values = get_inputs(input_output_pairs)
    output_values = get_outputs(input_output_pairs)
    n_rfms, n_data = size(output_values)

    input_dim = size(input_values, 1)

    kernel_structure = get_kernel_structure(srfi)
    n_hp = calculate_n_hyperparameters(input_dim, kernel_structure)


    rfms = get_rfms(srfi)
    if length(rfms) > 0
        @warn "ScalarRandomFeatureInterface already built. skipping..."
        return
    end
    fitted_features = get_fitted_features(srfi)
    n_features = get_n_features(srfi)
    batch_sizes = get_batch_sizes(srfi)
    rng = get_rng(srfi)
    decomp_type = get_feature_decomposition(srfi)
    optimizer_options = get_optimizer_options(srfi)
    optimizer = get_optimizer(srfi) # empty vector

    # Optimize features with EKP for each output dim
    # [1.] Split data into test/train 80/20 
    idx_shuffle = randperm(rng, n_data)
    n_cross_val_sets = Int(optimizer_options["n_cross_val_sets"])
    n_features_opt = optimizer_options["n_features_opt"]

    train_idx = []
    test_idx = []
    n_train = 0
    n_test = 0
    if n_cross_val_sets == 0
        push!(train_idx, idx_shuffle)
        push!(test_idx, idx_shuffle)
        n_cross_val_sets = 1 # now just pretend there is one partition for looping purposes
        n_train = n_data
        n_test = n_data
    else
        train_fraction = optimizer_options["train_fraction"]
        n_train = Int(floor(train_fraction * n_data))
        n_test = n_data - n_train

        if n_test * n_cross_val_sets > n_data
            throw(
                ArgumentError(
                    "train/test split produces cross validation test sets of size $(n_test), out of $(n_data). \"n_cross_val_sets\" optimizer_options keyword < $(Int(floor(n_data/n_test))). Received $n_cross_val_sets",
                ),
            )
        end


        for i in 1:n_cross_val_sets
            tmp = idx_shuffle[((i - 1) * n_test + 1):(i * n_test)]
            push!(test_idx, tmp)
            push!(train_idx, setdiff(collect(1:n_data), tmp))
        end
    end

    #regularization = I #
    regularization = isnothing(output_structure_matrix) ? 1.0 * I : Diagonal(output_structure_matrix)


    @info (
        "hyperparameter learning for $n_rfms models using $n_train training points, $n_test validation points and $n_features_opt features"
    )
    n_iteration = optimizer_options["n_iteration"]
    diagnostics = zeros(n_iteration, n_rfms)
    for i in 1:n_rfms
        regularization_i = regularization[i, i] * I

        io_pairs_opt = PairedDataContainer(input_values, reshape(output_values[i, :], 1, size(output_values, 2)))

        multithread = optimizer_options["multithread"]
        if multithread == "ensemble"
            multithread_type = EnsembleThreading()
        elseif multithread == "tullio"
            multithread_type = TullioThreading()
        else
            throw(
                ArgumentError(
                    "Unknown optimizer option for multithreading, please choose from \"tullio\" (allows Tullio.jl to control threading in RandomFeatures.jl), or \"ensemble\" (threading is done over the ensemble)",
                ),
            )
        end
        # [2.] Estimate covariance at mean value
        prior = optimizer_options["prior"]

        # where prior space has changed we need to rebuild the priors
        if ndims(prior) > n_hp

            # comes from having a truncated output_dimension
            # TODO not really a truncation here, resetting to default
            @info "Original input space of dimension $(get_input_dim(srfi)) has been truncated to $(input_dim). \n Rebuilding prior... number of hyperparameters reduced from $(ndims(prior)) to $(n_hp)."
            prior = build_default_prior(input_dim, kernel_structure)

        end

        μ_hp = transform_unconstrained_to_constrained(prior, mean(prior))

        cov_sample_multiplier = optimizer_options["cov_sample_multiplier"]
        cov_correction = optimizer_options["cov_correction"]
        n_cov_samples_min = n_test + 2
        n_cov_samples = Int(floor(n_cov_samples_min * max(cov_sample_multiplier, 0.0)))
        observation_vec = []
        for cv_idx in 1:n_cross_val_sets
            internal_Γ, approx_σ2 = estimate_mean_and_coeffnorm_covariance(
                srfi,
                rng,
                μ_hp,
                regularization_i,
                n_features_opt,
                train_idx[cv_idx],
                test_idx[cv_idx],
                batch_sizes,
                io_pairs_opt,
                n_cov_samples,
                decomp_type,
                multithread_type,
                cov_correction = cov_correction,
            )
            Γ = internal_Γ
            Γ[1:n_test, 1:n_test] += regularization_i  # + approx_σ2 
            Γ[(n_test + 1):end, (n_test + 1):end] += I
            if !isposdef(Γ)
                Γ = posdef_correct(Γ)
            end
            data = vcat(get_outputs(io_pairs_opt)[test_idx[cv_idx]], 0.0, 0.0)

            push!(
                observation_vec,
                EKP.Observation(Dict("names" => "$(cv_idx)", "samples" => data[:], "covariances" => Γ)),
            )
        end
        observation = combine_observations(observation_vec)
        # [3.] set up EKP optimization
        n_ensemble = optimizer_options["n_ensemble"]
        n_iteration = optimizer_options["n_iteration"]
        opt_verbose_flag = optimizer_options["verbose"]
        scheduler = optimizer_options["scheduler"]
        accelerator = optimizer_options["accelerator"]
        localization = optimizer_options["localization"]

        initial_params = construct_initial_ensemble(rng, prior, n_ensemble)
        ekiobj = EKP.EnsembleKalmanProcess(
            initial_params,
            observation,
            Inversion(),
            scheduler = scheduler,
            rng = rng,
            accelerator = accelerator,
            verbose = opt_verbose_flag,
            localization_method = localization,
        )
        err = zeros(n_iteration)

        # [4.] optimize with EKP
        for i in 1:n_iteration

            #get parameters:
            lvec = transform_unconstrained_to_constrained(prior, get_u_final(ekiobj))
            g_ens = zeros(n_cross_val_sets * (n_test + 2), n_ensemble)
            for cv_idx in 1:n_cross_val_sets

                g_ens_tmp, _ = calculate_ensemble_mean_and_coeffnorm(
                    srfi,
                    rng,
                    lvec,
                    regularization_i,
                    n_features_opt,
                    train_idx[cv_idx],
                    test_idx[cv_idx],
                    batch_sizes,
                    io_pairs_opt,
                    decomp_type,
                    multithread_type,
                )
                g_ens[((cv_idx - 1) * (n_test + 2) + 1):(cv_idx * (n_test + 2)), :] = g_ens_tmp
            end

            inflation = optimizer_options["inflation"]
            if inflation > 0
                terminated = EKP.update_ensemble!(ekiobj, g_ens, additive_inflation = true, s = inflation) # small regularizing inflation
            else
                terminated = EKP.update_ensemble!(ekiobj, g_ens) # small regularizing inflation
            end
            if !isnothing(terminated)
                break # if the timestep was terminated due to timestepping condition
            end

            err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
        end
        diagnostics[:, i] = copy(err)

        # [5.] extract optimal hyperparameters
        hp_optimal = get_ϕ_mean_final(prior, ekiobj)[:]

        if opt_verbose_flag
            names = get_name(prior)
            hp_optimal_batch = [hp_optimal[b] for b in batch(prior)]
            hp_optimal_range =
                [(minimum(hp_optimal_batch[i]), maximum(hp_optimal_batch[i])) for i in 1:length(hp_optimal_batch)] #the min and max of the hparams
            prior_conf_interval = [mean(prior) .- 3 * sqrt.(var(prior)), mean(prior) .+ 3 * sqrt.(var(prior))]
            pci_constrained = [transform_unconstrained_to_constrained(prior, prior_conf_interval[i]) for i in 1:2]
            pcic = [(pci_constrained[1][i], pci_constrained[2][i]) for i in 1:length(pci_constrained[1])]
            pcic_batched = [pcic[b][1] for b in batch(prior)]
            @info("EKI Optimization result:")
            println(
                display(
                    [
                        "name" "number of hyperparameters" "optimized value range" "99% prior mass"
                        names length.(hp_optimal_batch) hp_optimal_range pcic_batched
                    ],
                ),
            )
        end

        io_pairs_i = PairedDataContainer(input_values, reshape(output_values[i, :], 1, size(output_values, 2)))
        # Now, fit new RF model with the optimized hyperparameters

        rfm_i = RFM_from_hyperparameters(
            srfi,
            rng,
            hp_optimal,
            regularization_i,
            n_features,
            batch_sizes,
            input_dim,
            multithread_type,
        )
        fitted_features_i = RF.Methods.fit(rfm_i, io_pairs_i, decomposition_type = decomp_type) #fit features

        push!(rfms, rfm_i)
        push!(fitted_features, fitted_features_i)

    end
    push!(get_regularization(srfi), regularization)
    push!(optimizer, diagnostics)

end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Empty method, as optimization takes place within the build_models stage
"""
function optimize_hyperparameters!(srfi::ScalarRandomFeatureInterface, args...; kwargs...)
    @info("Random Features already trained. continuing...")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Prediction of emulator mean at new inputs (passed in as columns in a matrix), and a prediction of the total covariance at new inputs equal to (emulator covariance + noise covariance). 
"""
function predict(
    srfi::ScalarRandomFeatureInterface,
    new_inputs::MM;
    multithread = "ensemble",
) where {MM <: AbstractMatrix}
    M = length(get_rfms(srfi))
    N_samples = size(new_inputs, 2)
    # Predicts columns of inputs: input_dim × N_samples
    μ = zeros(M, N_samples)
    σ2 = zeros(M, N_samples)
    if multithread == "ensemble"
        tullio_threading = false
    elseif multithread == "tullio"
        tullio_threading = true
    end

    for i in 1:M
        μ[i, :], σ2[i, :] = RF.Methods.predict(
            get_rfms(srfi)[i],
            get_fitted_features(srfi)[i],
            DataContainer(new_inputs),
            tullio_threading = tullio_threading,
        )
    end

    # add the noise contribution stored within the regularization
    reg = get_regularization(srfi)[1]
    reg_diag = isa(reg, UniformScaling) ? reg.λ * ones(M) : diag(reg)

    for i in 1:M
        σ2[i, :] .+= reg_diag[i]
    end

    return μ, σ2
end
