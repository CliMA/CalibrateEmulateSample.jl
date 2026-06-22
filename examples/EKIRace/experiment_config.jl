using Dates

########################################################################
###############  USER TOGGLE  #########################################
########################################################################
# Set EXPERIMENT to one of:
experiments = [:l63, :l96_const, :l96_vec, :l96_flux]
EXPERIMENT = experiments[4]

# Date identifying this calibration run — written by calibrate, read by
# emulate_sample and exp_to_leaderboard (so all stages stay in sync).
# PIN this to reproduce a specific run directory.
#calibrate_date = Date("2026-06-03", "yyyy-mm-dd")
calibrate_date = today()

########################################################################
###############  SHARED CONSTANTS  ####################################
########################################################################
method_cases = ["Inversion", "TransformInversion", "Unscented", "GaussNewtonInversion"]

method_names = [
    ("Inversion()", "EKI"),
    ("TransformInversion()", "ETKI"),
    ("GaussNewtonInversion()", "GNKI"),
    ("Unscented(prior)", "UKI"),
]

method_cases_key = Dict(
    "Inversion"            => "ces-eki-dmc",
    "TransformInversion"   => "ces-etki-dmc",
    "Unscented"            => "ces-uki-dmc",
    "GaussNewtonInversion" => "ces-iekf",
)

forcing_cases_key = Dict(
    "const-force" => "ensemble_results",
    "vec-force"   => "spatial_forcing_ensemble_results",
    "flux-force"  => "nn_forcing_ensemble_results",
)

########################################################################
###############  PER-CASE CONFIG  #####################################
########################################################################
# Important dials:
#    terminate_at: psuedotime to terminate. (T=1 is approx posterior for the finite-time method variants)
#    N_iter: otherwise, maximum iterations
#    N_ens_sizes: Vector of experiments
#    n_repeats: number of rng seeds

function experiment_config(case::Symbol)
    n_ens_step = 8
    n_repeats = 20
    if case == :l63
        ens_step = 2
        return (
            model          = "l63",
            force_case     = nothing,
            N_ens_sizes    = collect(4:ens_step:4+n_ens_step*ens_step),
            N_iter         = 20,
            terminate_at   = 2.0,   # DataMisfitController end time
            n_repeats      = n_repeats,
            max_iter       = 10,
            retain_var     = 0.99,
            n_features     = 100,
            n_features_opt = 60,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_const
        ens_step_const = 2
        return (
            model          = "l96",
            force_case     = "const-force",
            N_ens_sizes    = collect(4:ens_step_const:4+n_ens_step*ens_step_const),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_vec
        ens_step_vec = 5
        return (
            model          = "l96",
            force_case     = "vec-force",
            N_ens_sizes    = collect(40:ens_step_vec:40+n_ens_step*ens_step_vec),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    elseif case == :l96_flux
        ens_step_flux = 5
        return (
            model          = "l96",
            force_case     = "flux-force",
            N_ens_sizes    = collect(30:ens_step_flux:30+n_ens_step*ens_step_flux),
            N_iter         = 20,
            terminate_at   = 2.0,
            n_repeats      = n_repeats,
            max_iter       = 15,
            retain_var     = 0.99,
            n_features     = 200,
            n_features_opt = 160,
            calibrate_date = calibrate_date,
        )
    else
        throw(ArgumentError("Unknown experiment: $case. Expected one of :l63, :l96_const, :l96_vec, :l96_flux"))
    end
end

########################################################################
###############  FILENAME BUILDERS  ###################################
########################################################################
function case_suffix(cfg, N_ens, rng_idx)
    if cfg.force_case === nothing
        return "$(N_ens)_$(rng_idx)"
    else
        return "$(cfg.force_case)_$(N_ens)_$(rng_idx)"
    end
end

calib_directory(method, cfg) = "$(method)_$(cfg.calibrate_date)"

function prior_filename(cfg)
    if cfg.force_case === nothing
        return "$(cfg.model)_priors.jld2"
    else
        return "$(cfg.model)_priors_$(cfg.force_case).jld2"
    end
end

ekp_filename(cfg, N_ens, rng_idx)      = "$(cfg.model)_ekp_$(case_suffix(cfg, N_ens, rng_idx)).jld2"
results_filename(cfg, N_ens, rng_idx)  = "$(cfg.model)_calibrate_results_$(case_suffix(cfg, N_ens, rng_idx)).jld2"
posterior_filename(cfg, N_ens, rng_idx) = "$(cfg.model)_posterior_$(case_suffix(cfg, N_ens, rng_idx)).jld2"

function summary_filename(cfg)
    if cfg.force_case === nothing
        return "$(cfg.model)_output_$(cfg.calibrate_date).jld2"
    else
        return "$(cfg.model)_calibrate_$(cfg.force_case)_$(cfg.calibrate_date).jld2"
    end
end

function nc_filename(cfg, method)
    key = method_cases_key[method]
    if cfg.force_case === nothing
        return "$(key)_$(cfg.model)_ensemble_results_$(cfg.calibrate_date).nc"
    else
        return "$(key)_$(cfg.model)_$(forcing_cases_key[cfg.force_case])_$(cfg.calibrate_date).nc"
    end
end

