# CONFIGURE THE THREE STEPS

## -- Configure the inverse problem --
problem = "lorenz" # "lorenz" or "linear" or "linear_exp" or "linlinexp"
input_dim = 40
output_dim = 80

## -- Configure parameters of the experiment itself --
rng_seed = 41
num_trials = 1
αs = 0.0:0.25:1.0
grad_types = (:perfect, :mean, :linreg, :localsl) # Out of :perfect, :mean, :linreg, and :localsl
Vgrad_types = () # Out of :egi

# Specific to step 1
step1_eki_ensemble_size = 200
step1_mcmc_sampler = :rw # :rw or :mala
step1_mcmc_samples_per_chain = 5_000
step1_mcmc_num_chains = 8
step1_mcmc_subsample_rate = 100

# Specific to step 2
step2_manopt_num_dims = 0
step2_Vgrad_num_samples = 8
step2_egi_ξ = 0.0
step2_egi_γ = 1.5

# Specific to step 3
step3_diagnostics_to_use =
    vcat([
        ("Hu_1.0_mcmc_$grad_type", i, "Hg_1.0_ekp_perfect", 80) for grad_type in grad_types for i in 4:2:16
    ], [
        ("Hu_$(α)_mcmc_perfect", i, "Hg_1.0_ekp_perfect", 80) for α in αs[1:end-1] for i in 4:2:16
    ], [
        ("pca_u", i, "Hg_1.0_ekp_perfect", 80) for i in 4:2:16
    ])
step3_run_reduced_in_full_space = false
step3_marginalization = :forward_model # :loglikelihood or :forward_model
step3_num_marginalization_samples = 1
step3_posterior_sampler = :mcmc # :eks or :mcmc
step3_eks_ensemble_size = 800 # only used if `step3_posterior_sampler == :eks`
step3_eks_max_iters = 200 # only used if `step3_posterior_sampler == :eks`
step3_mcmc_sampler = :rw # :rw or :mala; only used if `step3_posterior_sampler == :mcmc`
step3_mcmc_samples_per_chain = 2_000 # only used if `step3_posterior_sampler == :mcmc`
step3_mcmc_num_chains = 24 # only used if `step3_posterior_sampler == :mcmc`
