using Pkg
using Conda

# this is run whenever Pkg.add/instantiate/build is called
Conda.add("python=3.11", channel="conda-forge")
Conda.add("scipy=1.14.1", channel = "conda-forge")
Conda.add("scikit-learn=1.5.1")

using PyCall
if lowercase(get(ENV, "CI", "false")) == "true"
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")
end

