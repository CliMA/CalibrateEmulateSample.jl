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

Currently CES depends on some python dependencies including ScikitLearn.jl which requires a couple extra installation steps:

First clone the project into a new local repository

```
git clone git@github.com:Clima/CalibrateEmulateSample.jl
cd CalibrateEmulateSample.jl
```

Install all the project dependencies.  Given that CES depends on python packages it is easiest to set the project to use it's own conda environment

### Project Mind map
The latest code structure is found at:
https://miro.com/app/board/o9J_kkm8OGU=/
