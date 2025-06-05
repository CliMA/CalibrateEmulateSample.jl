# CONFIGURE THE THREE STEPS

## -- Configure the inverse problem --
problem = "linear_exp" # "lorenz" or "linear_exp"
input_dim = 50
output_dim = 50

## -- Configure parameters of the experiment itself --
rng_seed = 41
num_trials = 2

# Specific to step 1
step1_eki_ensemble_size = 800
step1_eki_max_iters = 20

# Specific to step 2
step2_num_prior_samples = 2000 # paper uses 5e5

# Specific to step 3
step3_diagnostics_to_use = [
    ("Hu", 50, "Hg", 50),
]
step3_run_reduced_in_full_space = true
step3_posterior_sampler = :eks # :eks or :mcmc
step3_eks_ensemble_size = 800 # only used if `step3_posterior_sampler == :eks`
step3_eks_max_iters = 200 # only used if `step3_posterior_sampler == :eks`
step3_mcmc_sampler = :rw # :rw or :mala; only used if `step3_posterior_sampler == :mcmc`
step3_mcmc_samples_per_chain = 50_000 # only used if `step3_posterior_sampler == :mcmc`
