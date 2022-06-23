# reference in tree version of CalibrateEmulateSample

using CalibrateEmulateSample
using Documenter
# using DocumenterCitations

#----------

examples = ["Lorenz example" => "examples/lorenz_example.md"]

design = ["AbstractMCMC sampling API" => "API/AbstractMCMC.md"]

api = [
    "CalibrateEmulateSample" => [
        "Emulators" => ["General Emulator" => "API/Emulators.md", "Gaussian Process" => "API/GaussianProcess.md"],
        "MarkovChainMonteCarlo" => "API/MarkovChainMonteCarlo.md",
        "Utilities" => "API/Utilities.md",
    ],
]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Contributing" => "contributing.md",
    "Emulator" => "emulate.md",
    "Examples" => examples,
    "Package Design" => design,
    "API" => api,
    "Glossary" => "glossary.md",
]

#----------

format = Documenter.HTML(collapselevel = 1, prettyurls = !isempty(get(ENV, "CI", "")))

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
