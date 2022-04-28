# CalibrateEmulateSample.jl

|||
|---------------------:|:----------------------------------------------|
| **Documentation**    | [![dev][docs-dev-img]][docs-dev-url]          |
| **Code Coverage**    | [![codecov][codecov-img]][codecov-url]        |
| **Bors**             | [![Bors enabled][bors-img]][bors-url]         |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://CliMA.github.io/CalibrateEmulateSample.jl/dev/

[codecov-img]: https://codecov.io/gh/CliMA/CalibrateEmulateSample.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/CliMA/CalibrateEmulateSample.jl

[bors-img]: https://bors.tech/images/badge_small.svg
[bors-url]: https://app.bors.tech/repositories/24774

## Installing CES

Currently CES depends on some python dependencies via ScikitLearn.jl which requires a couple of extra installation steps:

First clone the project into a new local repository

```
git clone git@github.com:Clima/CalibrateEmulateSample.jl
cd CalibrateEmulateSample.jl
```

Install all the project dependencies (see the documentation for specific details).  Given that CES depends on python packages it is easiest to set the project to use its own conda environment


## Building CES Documentation

You need to first build the toplevel project before building the documentation:

```
cd CalibrateEmulateSample.jl
julia --project -e 'using Pkg; Pkg.instantiate()
```

Then you can build the project documentation under the `docs/` sub-project:

```
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'
julia --project=docs/ docs/make.jl
```

The locally rendered HTML documentation can be viewed at `docs/build/index.html`

### Project Mind map
The rough code structure, and its interaction with EnsembleKalmanProcesses.jl is found at:
https://miro.com/app/board/o9J_kkm8OGU=/
