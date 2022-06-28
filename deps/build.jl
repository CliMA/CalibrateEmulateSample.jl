using Pkg
using PyCall
using Conda

if lowercase(get(ENV, "CI", "false")) == "true"
    ENV["PYTHON"] = ""
    Pkg.build("PyCall")
end

Conda.add("scikit-learn")
