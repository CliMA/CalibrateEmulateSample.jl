using CalibrateEmulateSample, Documenter

makedocs(
  sitename = "CalibrateEmulateSample.jl",
  doctest = false,
  strict = false,
  format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict()
            )
        ))
  ),
  clean = false,
  modules = [Documenter, CalibrateEmulateSample],
  pages = Any[
    "Home" => "index.md",
    "Ensemble Kalman Inversion" => "EKI.md",
    "Gaussian Process Emulator" => "GPEmulator.md",
    "Markov chain Monte-Carlo"  => "MCMC.md",
    "G-Model"                   => "GModel.md",
    "Observations"              => "Observations.md",
    "Utilities"                 => "Utilities.md",
  ],
)

deploydocs(
           repo = "github.com/climate-machine/CalibrateEmulateSample.jl.git",
           target = "build",
          )
