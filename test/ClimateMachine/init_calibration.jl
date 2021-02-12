using Distributions
using JLD
# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities


function generate_cm_params(cm_params::Union{Float64, Array{Float64}}, 
                            cm_param_names::Union{String,Array{String}})
    # Generate version
    version = rand(11111:99999)

    open("clima_param_defs_$(version).jl", "w") do io
        for i in 1:length(cm_params)
            write(io, "CLIMAParameters.Atmos.SubgridScale.
                $(cm_param_names[i])(::EarthParameterSet) = 
                $(cm_params[i])\n")
        end
    end
    return version
end

param_names = ["C_smag", "C_drag"]
n_param = length(param_names)
priors = [Priors.Prior(Distributions.Normal(0.25, 0.05), param_names[1]),
          Priors.Prior(Distributions.Normal(0.001, 0.0001), param_names[2])]

# Construct initial ensemble
N_ens = 10
initial_params = EKP.construct_initial_ensemble(N_ens, priors)
params_arr = [row[:] for row in eachrow(initial_params)]
versions = map(param -> generate_cm_params(param, param_names), params_arr)


open("versions.txt", "w") do io
        for version in versions
            write(io, "clima_param_defs_$(version).jl\n")
        end
    end
## Continue calibration
