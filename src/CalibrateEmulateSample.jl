module CalibrateEmulateSample

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# imported modules from EKP.
import EnsembleKalmanProcesses
EnsembleKalmanProcessModule = EnsembleKalmanProcesses.EnsembleKalmanProcessModule 
Observations = EnsembleKalmanProcesses.Observations
ParameterDistributionStorage = EnsembleKalmanProcesses.ParameterDistributionStorage
DataStorage = EnsembleKalmanProcesses.DataStorage


# No internal deps, heavy external deps
include("GaussianProcessEmulator.jl")

# Internal deps, light external deps
include("Utilities.jl")

# Internal deps, light external deps
include("MarkovChainMonteCarlo.jl")

end # module
