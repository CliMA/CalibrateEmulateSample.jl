using ProgressBars

export VectorRandomFeatureInterface
export get_rfms,
    get_fitted_features,
    get_batch_sizes,
    get_n_features,
    get_input_dim,
    get_output_dim,
    get_rng,
    get_diagonalize_input,
    get_diagonalize_output,
    get_optimizer_options

"""
$(DocStringExtensions.TYPEDEF)

Structure holding the Vector Random Feature models. 

# Fields
$(DocStringExtensions.TYPEDFIELDS)

"""
struct VectorRandomFeatureInterface{S <: AbstractString, RNG <: AbstractRNG} <: RandomFeatureInterface
    "A vector of `RandomFeatureMethod`s, contains the feature structure, batch-sizes and regularization"
    rfms::Vector{RF.Methods.RandomFeatureMethod}
    "vector of `Fit`s, containing the matrix decomposition and coefficients of RF when fitted to data"
    fitted_features::Vector{RF.Methods.Fit}
    "batch sizes"
    batch_sizes::Union{Dict{S, Int}, Nothing}
    "number of features"
    n_features::Union{Int, Nothing}
    "input dimension"
    input_dim::Int
    "output_dimension"
    output_dim::Int
    "rng"
    rng::RNG
    "regularization"
    regularization::Vector{Union{Matrix, UniformScaling, Diagonal}}
    "Assume inputs are decorrelated for ML"
    diagonalize_input::Bool
    "Assume outputs are decorrelated for ML. Note: emulator(..., decorrelate=true) by default"
    diagonalize_output::Bool
    "Random Feature decomposition, choose from \"svd\" or \"cholesky\" (default)"
    feature_decomposition::S
    "dictionary of options for hyperparameter optimizer"
    optimizer_options::Dict
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the rfms field
"""
get_rfms(vrfi::VectorRandomFeatureInterface) = vrfi.rfms

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the fitted_features field
"""
get_fitted_features(vrfi::VectorRandomFeatureInterface) = vrfi.fitted_features

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the batch_sizes field
"""
get_batch_sizes(vrfi::VectorRandomFeatureInterface) = vrfi.batch_sizes

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the n_features field
"""
get_n_features(vrfi::VectorRandomFeatureInterface) = vrfi.n_features

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the input_dim field
"""
get_input_dim(vrfi::VectorRandomFeatureInterface) = vrfi.input_dim

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the output_dim field
"""
get_output_dim(vrfi::VectorRandomFeatureInterface) = vrfi.output_dim

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the rng field
"""
get_rng(vrfi::VectorRandomFeatureInterface) = vrfi.rng

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the regularization field
"""
get_regularization(vrfi::VectorRandomFeatureInterface) = vrfi.regularization

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the diagonalize_input field
"""
get_diagonalize_input(vrfi::VectorRandomFeatureInterface) = vrfi.diagonalize_input

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the diagonalize_output field
"""
get_diagonalize_output(vrfi::VectorRandomFeatureInterface) = vrfi.diagonalize_output

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the feature_decomposition field
"""
get_feature_decomposition(vrfi::VectorRandomFeatureInterface) = vrfi.feature_decomposition

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the optimizer_options field
"""
get_optimizer_options(vrfi::VectorRandomFeatureInterface) = vrfi.optimizer_options

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructs a `VectorRandomFeatureInterface <: MachineLearningTool` interface for the `RandomFeatures.jl` package for multi-input and multi-output emulators.
 - `n_features` - the number of random features
 - `input_dim` - the dimension of the input space
 - `output_dim` - the dimension of the output space
 - `diagonalize_input = false` - whether to assume independence in input space
 - `diagonalize_output::Bool = false` - whether to assume independence in output space (even if set true, this is not the same as a list of scalar-valued models, as hyperparameters will be shared between models in this case)
 - `batch_sizes = nothing` - Dictionary of batch sizes passed `RandomFeatures.jl` object (see definition there)
 - `rng = Random.GLOBAL_RNG` - random number generator 
 - `feature_decomposition = "cholesky"` - choice of how to store decompositions of random features, `cholesky` or `svd` available
 - `optimizer_options = nothing` - Dict of options to pass into EKI optimization of hyperparameters (defaults created in `VectorRandomFeatureInterface` constructor):
      - "prior": the prior for the hyperparameter optimization
      - "prior_in_scale"/"prior_out_scale": use these to tune the input/output prior scale.
      - "n_ensemble": number of ensemble members
      - "n_iteration": number of eki iterations
      - "scheduler": Learning rate Scheduler (a.k.a. EKP timestepper) Default: DataMisfitController
      - "cov_sample_multiplier": increase for more samples to estimate covariance matrix in optimization (default 10.0, minimum 1.0) 
      - "tikhonov": tikhonov regularization parameter if > 0
      - "inflation": additive inflation ∈ [0,1] with 0 being no inflation
      - "train_fraction": e.g. 0.8 (default)  means 80:20 train - test split
      - "multithread": how to multithread. "ensemble" (default) threads across ensemble members "tullio" threads random feature matrix algebra
      - "verbose" => false, verbose optimizer statements to check convergence, priors and optimal parameters.

"""
function VectorRandomFeatureInterface(
    n_features::Int,
    input_dim::Int,
    output_dim::Int;
    diagonalize_input::Bool = false,
    diagonalize_output::Bool = false,
    batch_sizes::Union{Dict{S, Int}, Nothing} = nothing,
    rng::RNG = Random.GLOBAL_RNG,
    feature_decomposition::S = "cholesky",
    optimizer_options::Union{Dict{S}, Nothing} = nothing,
) where {S <: AbstractString, RNG <: AbstractRNG}

    # Initialize vector for RF model
    rfms = Vector{RF.Methods.RandomFeatureMethod}(undef, 0)
    fitted_features = Vector{RF.Methods.Fit}(undef, 0)
    regularization = Vector{Union{Matrix, UniformScaling, Nothing}}(undef, 0)

    #Optimization Defaults    
    if !isnothing(optimizer_options)
        prior_in_scale = "prior_in_scale" ∈ keys(optimizer_options) ? optimizer_options["prior_in_scale"] : 1.0
        prior_out_scale = "prior_out_scale" ∈ keys(optimizer_options) ? optimizer_options["prior_out_scale"] : 1.0
    else
        prior_in_scale = 1.0
        prior_out_scale = 1.0
    end
    prior = build_default_prior(
        input_dim,
        output_dim,
        diagonalize_input,
        diagonalize_output,
        in_scale = prior_in_scale,
        out_scale = prior_out_scale,
    )
    optimizer_opts = Dict(
        "prior" => prior, #the hyperparameter_prior (note scalings have already been applied)
        "n_ensemble" => max(ndims(prior) + 1, 10), #number of ensemble
        "n_iteration" => 5, # number of eki iterations
        "scheduler" => EKP.DataMisfitController(), # Adaptive timestepping
        "cov_sample_multiplier" => 10.0, # multiplier for samples to estimate covariance in optimization scheme 
        "tikhonov" => 0, # tikhonov regularization parameter if >0
        "inflation" => 1e-4, # additive inflation ∈ [0,1] with 0 being no inflation
        "train_fraction" => 0.8, # 80:20 train - test split
        "multithread" => "ensemble", # instead of "tullio"
        "verbose" => false, # verbose optimizer statements
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

    return VectorRandomFeatureInterface{S, RNG}(
        rfms,
        fitted_features,
        batch_sizes,
        n_features,
        input_dim,
        output_dim,
        rng,
        regularization,
        diagonalize_input,
        diagonalize_output,
        feature_decomposition,
        optimizer_opts,
    )
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the random feature method from hyperparameters. We use cosine activation functions and a Matrixvariate Normal distribution with mean M=0, and input(output) covariance U(V) built from cholesky factorization or diagonal components based on the diagonalize_input(diagonalize_output) flag.
"""
function RFM_from_hyperparameters(
    vrfi::VectorRandomFeatureInterface,
    rng::RNG,
    l::ForVM,
    regularization::MorUSorD,
    n_features::Int,
    batch_sizes::Union{Dict{S, Int}, Nothing},
    input_dim::Int,
    output_dim::Int,
    multithread_type::MT;
) where {
    S <: AbstractString,
    RNG <: AbstractRNG,
    ForVM <: Union{AbstractFloat, AbstractVecOrMat},
    MorUSorD <: Union{Matrix, UniformScaling, Diagonal},
    MT <: MultithreadType,
}
    diagonalize_input = get_diagonalize_input(vrfi)
    diagonalize_output = get_diagonalize_output(vrfi)

    M = zeros(input_dim, output_dim) # n x p mean
    U, V = hyperparameters_from_flat(l, input_dim, output_dim, diagonalize_input, diagonalize_output)

    if !isposdef(U)
        println("U not posdef - correcting")
        U = posdef_correct(U)
    end
    if !isposdef(V)
        println("V not posdef - correcting")
        V = posdef_correct(V)
    end

    dist = MatrixNormal(M, U, V)
    pd = ParameterDistribution(
        Dict(
            "distribution" => Parameterized(dist),
            "constraint" => repeat([no_constraint()], input_dim * output_dim),
            "name" => "xi",
        ),
    )
    feature_sampler = RF.Samplers.FeatureSampler(pd, output_dim, rng = rng)
    vff = RF.Features.VectorFourierFeature(n_features, output_dim, feature_sampler)

    thread_opt = isa(multithread_type, TullioThreading) # if we want to multithread with tullio
    if isnothing(batch_sizes)
        return RF.Methods.RandomFeatureMethod(vff, regularization = regularization, tullio_threading = thread_opt)
    else
        return RF.Methods.RandomFeatureMethod(
            vff,
            regularization = regularization,
            batch_sizes = batch_sizes,
            tullio_threading = thread_opt,
        )
    end
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build Vector Random Feature model for the input-output pairs subject to regularization, and optimizes the hyperparameters with EKP. 
"""
function build_models!(
    vrfi::VectorRandomFeatureInterface,
    input_output_pairs::PairedDataContainer{FT};
    regularization_matrix::Union{M, Nothing} = nothing,
) where {FT <: AbstractFloat, M <: AbstractMatrix}

    # get inputs and outputs 
    input_values = get_inputs(input_output_pairs)
    output_values = get_outputs(input_output_pairs)
    n_rfms, n_data = size(output_values)

    input_dim = size(input_values, 1)
    output_dim = size(output_values, 1)
    # there are less hyperparameters when we diagonalize
    diagonalize_input = get_diagonalize_input(vrfi)
    diagonalize_output = get_diagonalize_output(vrfi)
    n_hp = calculate_n_hyperparameters(input_dim, output_dim, diagonalize_input, diagonalize_output)

    rfms = get_rfms(vrfi)
    fitted_features = get_fitted_features(vrfi)
    n_features = get_n_features(vrfi)
    batch_sizes = get_batch_sizes(vrfi)
    decomp_type = get_feature_decomposition(vrfi)
    optimizer_options = get_optimizer_options(vrfi)
    multithread = optimizer_options["multithread"]
    if multithread == "ensemble"
        multithread_type = EnsembleThreading()
    elseif multithread == "tullio"
        multithread_type = TullioThreading()
    else
        throw(
            ArgumentError(
                "Unknown optimizer option for multithreading, please choose from \"tullio\" (allows Tullio.jl to control threading in RandomFeatures.jl, or \"loops\" (threading optimization loops)",
            ),
        )
    end
    prior = optimizer_options["prior"]
    rng = get_rng(vrfi)

    if ndims(prior) > n_hp
        # comes from having a truncated output_dimension
        # TODO not really a truncation here, resetting to default
        @info "truncating hyperparameter space to default in first $n_hp dimensions, due to truncation of output space"
        prior = constrained_gaussian("cholesky_factors", 1.0, 1.0, 0.0, Inf, repeats = n_hp)

    end

    # Optimize feature cholesky factors with EKP
    # [1.] Split data into test/train (e.g. 80/20)
    train_fraction = optimizer_options["train_fraction"]
    n_train = Int(floor(train_fraction * n_data)) # 20% split
    n_test = n_data - n_train
    if diagonalize_output
        n_features_opt = min(n_features, Int(floor(4 * n_train))) #we take a specified number of features for optimization. MAGIC NUMBER
    else
        # note the n_features_opt should NOT exceed output_dim * n_train or the regularization is replaced with a diagonal approx.
        n_features_opt = max(min(n_features, Int(floor(4 * sqrt(n_train * output_dim)))), Int(floor(1.9 * n_train)))#we take a specified number of features for optimization. MAGIC NUMBER
    end
    @info (
        "hyperparameter learning using $n_train training points, $n_test validation points and $n_features_opt features"
    )


    # regularization_matrix = nothing when we use scaled SVD to decorrelate the space,
    # in this setting, noise = I
    if regularization_matrix === nothing
        regularization = I
    else
        reg_mat = regularization_matrix
        if !isposdef(regularization_matrix)
            reg_mat = posdef_correct(regularization_matrix)
            println("RF regularization matrix is not positive definite, correcting")

        else
            # think of the regularization_matrix as the observational noise covariance, or a related quantity
            regularization = exp((1 / output_dim) * sum(log.(eigvals(reg_mat)))) * I #i.e. det(M)^{1/output_dim} I

            #regularization = reg_mat #using the full p.d. tikhonov exp. EXPENSIVE, and challenge get complexity terms
        end
    end


    idx_shuffle = randperm(rng, n_data)

    io_pairs_opt = PairedDataContainer(
        input_values[:, idx_shuffle],
        reshape(output_values[:, idx_shuffle], :, size(output_values, 2)),
    )

    # [2.] Estimate covariance at mean value
    μ_hp = transform_unconstrained_to_constrained(prior, mean(prior))
    cov_sample_multiplier = optimizer_options["cov_sample_multiplier"]

    if diagonalize_output
        n_cov_samples_min = n_test + 2 # diagonal case
    else
        n_cov_samples_min = (n_test * output_dim + 2) # min required for
    end
    n_cov_samples = Int(floor(n_cov_samples_min * max(cov_sample_multiplier, 1.0)))

    internal_Γ, approx_σ2 = estimate_mean_and_coeffnorm_covariance(
        vrfi,
        rng,
        μ_hp, # take mean values
        regularization,
        n_features_opt,
        n_train,
        n_test,
        batch_sizes,
        io_pairs_opt,
        n_cov_samples,
        decomp_type,
        multithread_type,
    )

    tikhonov_opt_val = optimizer_options["tikhonov"]
    if tikhonov_opt_val == 0
        # Build the covariance
        Γ = internal_Γ
        Γ[1:(n_test * output_dim), 1:(n_test * output_dim)] += approx_σ2
        Γ[(n_test * output_dim + 1):end, (n_test * output_dim + 1):end] += I

        #in diag case we have data logdet = λ^m, in non diag case we have logdet(Λ^) to match the different reg matrices.
        min_complexity =
            isa(regularization, UniformScaling) ? n_features_opt * log(regularization.λ) :
            n_features_opt / output_dim * 2 * sum(log.(diag(cholesky(regularization).L)))

        data = vcat(reshape(get_outputs(io_pairs_opt)[:, (n_train + 1):end], :, 1), 0.0, min_complexity) #flatten data

    elseif tikhonov_opt_val > 0
        # augment the state to add tikhonov
        outsize = size(internal_Γ, 1)
        Γ = zeros(outsize + n_hp, outsize + n_hp)
        Γ[1:outsize, 1:outsize] = internal_Γ
        Γ[1:(n_test * output_dim), 1:(n_test * output_dim)] += approx_σ2
        Γ[(n_test * output_dim + 1):outsize, (n_test * output_dim + 1):outsize] += I

        Γ[(outsize + 1):end, (outsize + 1):end] = tikhonov_opt_val .* cov(prior)

        #TODO the min complexity here is not the correct object in the non-diagonal case
        min_complexity =
            isa(regularization, UniformScaling) ? n_features_opt * log(regularization.λ) :
            n_features_opt / output_dim * 2 * sum(log.(diag(cholesky(regularization).L)))

        data = vcat(
            reshape(get_outputs(io_pairs_opt)[:, (n_train + 1):end], :, 1),
            0.0,
            min_complexity,
            zeros(size(Γ, 1) - outsize, 1),
        ) #flatten data with additional zeros
    else
        throw(
            ArgumentError(
                "Tikhonov parameter must be non-negative, instead received tikhonov_opt_val=$tikhonov_opt_val",
            ),
        )
    end

    # [3.] set up EKP optimization
    n_ensemble = optimizer_options["n_ensemble"] # minimal ensemble size n_hp,
    n_iteration = optimizer_options["n_iteration"]
    opt_verbose_flag = optimizer_options["verbose"]
    scheduler = optimizer_options["scheduler"]
    if !isposdef(Γ)
        Γ = posdef_correct(Γ)
    end

    initial_params = construct_initial_ensemble(rng, prior, n_ensemble)

    ekiobj = EKP.EnsembleKalmanProcess(
        initial_params,
        data[:],
        Γ,
        Inversion(),
        scheduler = scheduler,
        rng = rng,
        verbose = opt_verbose_flag,
    )
    err = zeros(n_iteration)

    # [4.] optimize with EKP
    for i in 1:n_iteration
        #get parameters:
        lvec = get_ϕ_final(prior, ekiobj)

        g_ens, _ = calculate_ensemble_mean_and_coeffnorm(
            vrfi,
            rng,
            lvec,
            regularization,
            n_features_opt,
            n_train,
            n_test,
            batch_sizes,
            io_pairs_opt,
            decomp_type,
            multithread_type,
        )
        if tikhonov_opt_val > 0
            # augment with the computational parameters (u not ϕ)
            uvecormat = get_u_final(ekiobj)
            if isa(uvecormat, AbstractVector)
                umat = reshape(uvecormat, 1, :)
            else
                umat = uvecormat
            end

            g_ens = vcat(g_ens, umat)
        end
        inflation = optimizer_options["inflation"]
        if inflation > 0
            terminated =
                EKP.update_ensemble!(ekiobj, g_ens, additive_inflation = true, use_prior_cov = true, s = inflation) # small regularizing inflation
        else
            terminated = EKP.update_ensemble!(ekiobj, g_ens) # small regularizing inflation
        end
        if !isnothing(terminated)
            break # if the timestep was terminated due to timestepping condition
        end

        err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)

    end

    # [5.] extract optimal hyperparameters
    hp_optimal = get_ϕ_mean_final(prior, ekiobj)[:]
    # Now, fit new RF model with the optimized hyperparameters
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
    rfm_tmp = RFM_from_hyperparameters(
        vrfi,
        rng,
        hp_optimal,
        regularization,
        n_features,
        batch_sizes,
        input_dim,
        output_dim,
        multithread_type,
    )
    fitted_features_tmp = fit(rfm_tmp, input_output_pairs, decomposition_type = decomp_type)


    push!(rfms, rfm_tmp)
    push!(fitted_features, fitted_features_tmp)
    push!(get_regularization(vrfi), regularization)

end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Empty method, as optimization takes place within the build_models stage
"""
function optimize_hyperparameters!(vrfi::VectorRandomFeatureInterface, args...; kwargs...)
    @info("Random Features already trained. continuing...")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Prediction of data observation (not latent function) at new inputs (passed in as columns in a matrix). That is, we add the observational noise into predictions.
"""
function predict(vrfi::VectorRandomFeatureInterface, new_inputs::M) where {M <: AbstractMatrix}
    input_dim = get_input_dim(vrfi)
    output_dim = get_output_dim(vrfi)
    rfm = get_rfms(vrfi)[1]
    ff = get_fitted_features(vrfi)[1]
    optimizer_options = get_optimizer_options(vrfi)
    multithread = optimizer_options["multithread"]
    if multithread == "ensemble"
        tullio_threading = false
    elseif multithread == "tullio"
        tullio_threading = true
    end

    N_samples = size(new_inputs, 2)
    # Predicts columns of inputs: input_dim × N_samples
    μ, σ2 = RF.Methods.predict(rfm, ff, DataContainer(new_inputs), tullio_threading = tullio_threading)
    # sizes (output_dim x n_test), (output_dim x output_dim x n_test) 
    # add the noise contribution from the regularization
    # note this is because we are predicting the data here, not the latent function.
    lambda = get_regularization(vrfi)[1]
    for i in 1:N_samples
        σ2[:, :, i] = 0.5 * (σ2[:, :, i] + permutedims(σ2[:, :, i], (2, 1))) + lambda

        if !isposdef(σ2[:, :, i])
            σ2[:, :, i] = posdef_correct(σ2[:, :, i])
        end
    end

    return μ, σ2
end
