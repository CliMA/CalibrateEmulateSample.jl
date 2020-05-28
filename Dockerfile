FROM julia:1.3

COPY . /CalibrateEmulateSample.jl/

WORKDIR /CalibrateEmulateSample.jl/
RUN julia --project -e "using Pkg; Pkg.instantiate();"
