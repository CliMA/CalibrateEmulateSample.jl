using Pkg
using PyCall
using Conda

if lowercase(get(ENV, "CI", "false")) == "true"
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")
end

Conda.add("scipy=1.8.1")
Conda.add("scikit-learn=1.1.1")
