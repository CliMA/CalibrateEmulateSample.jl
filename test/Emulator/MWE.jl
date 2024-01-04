using PyCall
using ScikitLearn
const pykernels = PyNULL()
const pyGP = PyNULL()

function init()
    copy!(pykernels, pyimport_conda("sklearn.gaussian_process.kernels", "scikit-learn=1.3.2"))
    copy!(pyGP, pyimport_conda("sklearn.gaussian_process", "scikit-learn=1.3.2"))
end

function minimal_failing_example()
    # some kind of kernel
    kernel = pykernels.ConstantKernel(constant_value = 1.0)
    
    # some data
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), 1, n)         # predictors/features: 1 x n
    y = reshape(sin.(x) + 0.05 * randn(n)', 1, n) # predictands/targets: 1 x n

    # the model
    m = pyGP.GaussianProcessRegressor(
        kernel = kernel,
    )

    # call fit!
    ScikitLearn.fit!(m, x, y)
    @info "fit successful"
end

init()
minimal_failing_example()
