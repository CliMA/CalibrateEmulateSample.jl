using PyCall
using ScikitLearn
const pykernels = PyNULL()
const pyGP = PyNULL()

function init()
    copy!(pykernels, pyimport_conda("sklearn.gaussian_process.kernels", "scikit-learn=1.1.1"))
    copy!(pyGP, pyimport_conda("sklearn.gaussian_process", "scikit-learn=1.1.1"))
end

function minimal_failing_example()
    # some kind of kernel
    var = pykernels.ConstantKernel(constant_value = 1.0)
    se = pykernels.RBF(1.0)
    white_noise_level = 1.0
    white = pykernels.WhiteKernel(
        noise_level = white_noise_level,
        noise_level_bounds = (1e-05, 10.0)
    )
    kernel = var * se + white

    # alg parameters
    alpha = 1e-3
    
    # some data
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), 1, n)         # predictors/features: 1 x n
    y = reshape(sin.(x) + 0.05 * randn(n)', 1, n) # predictands/targets: 1 x n

    # the model
    m = pyGP.GaussianProcessRegressor(
        kernel = kernel,
        n_restarts_optimizer = 10,
        alpha = alpha
    )

    # call fit!
    ScikitLearn.fit!(m, x, y)
    @info "fit successful"
end

init()
minimal_failing_example()
