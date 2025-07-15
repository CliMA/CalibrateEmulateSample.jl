# code deprecated due to JLD2
#=   
prior_filepath = joinpath(exp_dir, "prior.jld2")
if !isfile(prior_filepath)
    LoadError("prior file \"prior.jld2\" not found in directory \"" * exp_dir * "/\"")
else
    prior_dict_raw = load(prior_filepath) #using JLD2
    prior = prior_dict_raw["prior"]
end
=#
# build prior if jld2 does not work
function get_prior_config(s::SS) where {SS <: AbstractString}
    config = Dict()
    if s == "ent-det-calibration"
        config["constraints"] =
            Dict("entrainment_factor" => [bounded(0.0, 1.0)], "detrainment_factor" => [bounded(0.0, 1.0)])
        config["prior_mean"] = Dict("entrainment_factor" => 0.13, "detrainment_factor" => 0.51)
        config["unconstrained_σ"] = 1.0
    elseif s == "ent-det-tked-tkee-stab-calibration"
        config["constraints"] = Dict(
            "entrainment_factor" => [bounded(0.0, 1.0)],
            "detrainment_factor" => [bounded(0.0, 1.0)],
            "tke_ed_coeff" => [bounded(0.01, 1.0)],
            "tke_diss_coeff" => [bounded(0.01, 1.0)],
            "static_stab_coeff" => [bounded(0.01, 1.0)],
        )
        config["prior_mean"] = Dict(
            "entrainment_factor" => 0.13,
            "detrainment_factor" => 0.51,
            "tke_ed_coeff" => 0.14,
            "tke_diss_coeff" => 0.22,
            "static_stab_coeff" => 0.4,
        )
        config["unconstrained_σ"] = 1.0
    else
        throw(ArgumentError("prior for experiment $s not found, please implement in uq_for_edmf.jl"))
    end
    return config
end
