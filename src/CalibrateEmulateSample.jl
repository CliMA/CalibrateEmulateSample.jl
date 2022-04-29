"""
# Imported modules:
$(IMPORTS)

# Exports:
$(EXPORTS)
"""
module CalibrateEmulateSample

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# imported modules from EKP.
import EnsembleKalmanProcesses: EnsembleKalmanProcesses, ParameterDistributions, Observations, DataContainers

export EnsembleKalmanProcesses, ParameterDistributions, Observations, DataContainers

# No internal deps, heavy external deps
#include("GaussianProcessEmulator.jl")
include("Emulator.jl")

# Internal deps, light external deps
include("Utilities.jl")

# Internal deps, light external deps
include("MarkovChainMonteCarlo.jl")

end # module
