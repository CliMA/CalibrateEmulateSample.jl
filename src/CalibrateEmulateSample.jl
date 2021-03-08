module CalibrateEmulateSample

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

# imports from EKP.
import EnsembleKalmanProcesses

EnsembleKalmanProcessModule = EnsembleKalmanProcesses.EnsembleKalmanProcessModule 
Observations = EnsembleKalmanProcesses.Observations
ParameterDistributionStorage = EnsembleKalmanProcesses.ParameterDistributionStorage
DataStorage = EnsembleKalmanProcesses.DataStorage



# No internal deps, light external deps
#ParameterDistributionStorage = EnsembleKalmanProcessModule.ParameterDistributionStorage
#DataStorage = EnsembleKalmanProcessModule.DataStorage

#include("Observations.jl")
#include("ParameterDistribution.jl")
#include("DataStorage.jl")

# No internal deps, heavy external deps
include("GaussianProcessEmulator.jl")

# Internal deps, light external deps
include("Utilities.jl")

# Internal deps, light external deps
include("MarkovChainMonteCarlo.jl")

end # module
