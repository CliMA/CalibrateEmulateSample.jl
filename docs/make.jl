# reference in tree version of CalibrateEmulateSample

using CalibrateEmulateSample
using Documenter

#----------

examples = [
    "Simple example walkthrough" => "examples/sinusoid_example.md",
    "Lorenz example" => "examples/lorenz_example.md",
    "Turbulence example" => "examples/edmf_example.md",
    "Cloudy example" => "examples/Cloudy_example.md",
    "Emulator testing" => [
        "examples/emulators/regression_2d_2d.md",
        "examples/emulators/lorenz_integrator_3d_3d.md",
        "examples/emulators/global_sens_analysis.md",
    ],
]

api = [
    "CalibrateEmulateSample" => [
        "Emulators" => [
            "General Interface" => "API/Emulators.md",
            "Gaussian Process" => "API/GaussianProcess.md",
            "Random Features" => "API/RandomFeatures.md",
        ],
        "MarkovChainMonteCarlo" => "API/MarkovChainMonteCarlo.md",
        "Utilities" => "API/Utilities.md",
    ],
]

emulate = [
    "Emulator" => "emulate.md",
    "Gaussian Process" => "GaussianProcessEmulator.md",
    "Random Features" => "random_feature_emulator.md",
]
pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Contributing" => "contributing.md",
    "Examples" => examples,
    "Calibrate" => "calibrate.md",
    "Emulate" => emulate,
    "Sample" => "sample.md",
    "Glossary" => "glossary.md",
    "API" => api,
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
