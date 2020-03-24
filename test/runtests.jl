using Test
using CalibrateEmulateSample
ENV["JULIA_LOG_LEVEL"] = "WARN"
ENV["PYTHON"] = ""

for submodule in [
                  "EKI",
                  "GPEmulator",   # need unit tests
                  # "MCMC",         # need unit tests
                  "Observations", # need unit tests
                  # "GModel",       # need unit tests
                  # "Utilities",    # need unit tests
                  # "spaces",       # need unit tests
                  # "problems",     # need unit tests
                  # "Histograms",
                  "EKS",
                  # "GPR",          # need unit tests
                  #
                  # "L96m",
                  # "Cloudy",
                 ]

  println("Starting tests for $submodule")
  t = @elapsed include(joinpath(submodule,"runtests.jl"))
  println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end
