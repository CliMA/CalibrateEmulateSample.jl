using Test
using Solus
ENV["JULIA_LOG_LEVEL"] = "WARN"

include("neki.jl")

for submodule in ["L96m",
                  "GPRWrap",
                 ]

  println("Starting tests for $submodule")
  t = @elapsed include(joinpath(submodule,"runtests.jl"))
  println("Completed tests for $submodule, $(round(Int, t)) seconds elapsed")
end
