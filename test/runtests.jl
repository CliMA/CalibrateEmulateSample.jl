using Test

TEST_PLOT_OUTPUT = get(ENV, "CES_TEST_PLOT_OUTPUT", false)
@info "[in test/runtest.jl], create plots? CES_TEST_PLOT_OUTPUT: $(TEST_PLOT_OUTPUT)"

if TEST_PLOT_OUTPUT
    using Plots
end

# Python dependency versions (python, scipy, scikit-learn) are pinned in CondaPkg.toml
# and provisioned automatically via CondaPkg.jl / PythonCall.jl.

function include_test(_module)
    println("Starting tests for $_module")
    t = @elapsed include(joinpath(_module, "runtests.jl"))
    println("Completed tests for $_module, $(round(Int, t)) seconds elapsed")
    return nothing
end

@testset "CalibrateEmulateSample" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false

    function has_submodule(sm)
        any(ARGS) do a
            a == sm && return true
            first(split(a, '/')) == sm && return true
            return false
        end
    end

    for submodule in ["Emulator", "GaussianProcess", "RandomFeature", "MarkovChainMonteCarlo", "Utilities", "Show"]
        if all_tests || has_submodule(submodule) || "CalibrateEmulateSample" in ARGS
            include_test(submodule)
        end
    end
end
