# CONFIGURE THE THREE STEPS

## -- Configure the inverse problem --
problem = "linear" # "lorenz" or "linear" or "linear_exp"
input_dim = 200
output_dim = 50

## -- Configure parameters of the experiment itself --
rng_seed = 41
num_trials = 1

# Specific to step 1
step1_eki_ensemble_size = 800
step1_eki_max_iters = 20
step1_mcmc_temperature = 1.0 # 1.0 is the "true" posterior; higher oversamples the tails
step1_mcmc_sampler = :rw # :rw or :mala
step1_mcmc_samples_per_chain = 50_000
step1_mcmc_num_chains = 8
step1_mcmc_subsample_rate = 1000

# Specific to step 2
step2_num_prior_samples = 5_000 # paper uses 5e5

# Specific to step 3
step3_diagnostics_to_use = [
    ("Huy", 4, "Hg", 16),
    ("Huy", 8, "Hg", 16),
    ("Huy", 16, "Hg", 16),
]
step3_run_reduced_in_full_space = false
step3_marginalization = :forward_model # :loglikelihood or :forward_model
step3_num_marginalization_samples = 8
step3_posterior_sampler = :mcmc # :eks or :mcmc
step3_eks_ensemble_size = 800 # only used if `step3_posterior_sampler == :eks`
step3_eks_max_iters = 200 # only used if `step3_posterior_sampler == :eks`
step3_mcmc_sampler = :rw # :rw or :mala; only used if `step3_posterior_sampler == :mcmc`
step3_mcmc_samples_per_chain = 20_000 # only used if `step3_posterior_sampler == :mcmc`
step3_mcmc_num_chains = 8 # only used if `step3_posterior_sampler == :mcmc`
