# reference in tree version of CalibrateEmulateSample
prepend!(LOAD_PATH, [joinpath(@__DIR__, "..")])

using CalibrateEmulateSample
using Documenter
# using DocumenterCitations

#----------
api = ["CalibrateEmulateSample" => [
         "GPEmulator" => "API/GaussianProcessEmulator.md",
				 "MarkovChainMonteCarlo" => "API/MarkovChainMonteCarlo.md", 
				 "Utilities" => "API/Utilities.md"
         ],
]

examples = ["Lorenz example" => "examples/lorenz_example.md"]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Examples" => examples,
    "API" => api,
]

#----------

format = Documenter.HTML(
    collapselevel = 1,
    prettyurls = !isempty(get(ENV, "CI", ""))
)

makedocs(
  sitename = "CalibrateEmulateSample.jl",
  authors = "CliMA Contributors",
  format = format,
  pages = pages,
  modules = [CalibrateEmulateSample],
  doctest = false,
  strict = true,
  clean = true,
  checkdocs = :none,
)

if !isempty(get(ENV, "CI", ""))
  deploydocs(
    repo = "github.com/CliMA/CalibrateEmulateSample.jl.git",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
    push_preview = true,
  )
end
