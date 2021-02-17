module CalibrateEmulateSample

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# No internal deps, light external deps
include("Observations.jl")
include("Priors.jl")

# No internal deps, heavy external deps
include("EKP.jl")
#include("GPEmulator.jl")

# Internal deps, light external deps
include("Utilities.jl")

# Internal deps, light external deps
#include("MCMC.jl")

# Netcdf utils for clima
include("Netcdf_utils.jl")
end # module
