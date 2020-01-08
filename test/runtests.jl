using Test
using CalibrateEmulateSample
ENV["JULIA_LOG_LEVEL"] = "WARN"

include("eks.jl")

for submodule in ["L96m",
                  "GPR",
                  "Histograms",
                  "ConvenienceFunctions",
                 ]

  println("Starting tests for $submodule")
  t = @elapsed include(joinpath(submodule,"runtests.jl"))
  println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end
