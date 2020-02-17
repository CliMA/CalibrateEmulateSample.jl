using Test
using CalibrateEmulateSample
ENV["JULIA_LOG_LEVEL"] = "WARN"
ENV["PYTHON"] = ""

for submodule in [
                  "Utilities",
                  "L96m",
                  # "GPR",
                  "EKS",
                  # "Histograms",
                  "Cloudy",
                 ]

  println("Starting tests for $submodule")
  t = @elapsed include(joinpath(submodule,"runtests.jl"))
  println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end
