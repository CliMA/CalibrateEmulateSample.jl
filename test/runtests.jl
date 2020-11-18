using Test
using CalibrateEmulateSample

for submodule in [
                  "Priors",
                  "EKP",
                  "GPEmulator",
                  "MCMC",
                  "Observations",
                  "Utilities",
                 ]

  println("Starting tests for $submodule")
  t = @elapsed include(joinpath(submodule,"runtests.jl"))
  println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end

