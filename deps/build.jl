using Pkg
using PyCall
using Conda

if lowercase(get(ENV, "CI", "false")) == "true"
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")
end

Conda.add("scipy=1.14.1", channel = "conda-forge")
Conda.add("scikit-learn=1.5.1")
