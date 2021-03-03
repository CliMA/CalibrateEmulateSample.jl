using Distributions
using JLD
using ArgParse
# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.ClimaUtils

# Set parameter priors
param_names = ["C_smag", "C_drag"]
n_param = length(param_names)
priors = [Priors.Prior(Distributions.Normal(0.5, 0.05), param_names[1]),
          Priors.Prior(Distributions.Normal(0.001, 0.0001), param_names[2])]

# Construct initial ensemble
N_ens = 10
initial_params = EKP.construct_initial_ensemble(N_ens, priors)
# Generate CLIMAParameters files
params_arr = [row[:] for row in eachrow(initial_params)]
versions = map(param -> generate_cm_params(param, param_names), params_arr)

# Store version identifiers for this ensemble in a common file
open("versions_1.txt", "w") do io
        for version in versions
            write(io, "clima_param_defs_$(version).jl\n")
        end
    end
