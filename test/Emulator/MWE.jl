using PythonCall
const pykernels = PythonCall.pynew()
const pyGP = PythonCall.pynew()

function init()
    PythonCall.pycopy!(pykernels, pyimport("sklearn.gaussian_process.kernels"))
    PythonCall.pycopy!(pyGP, pyimport("sklearn.gaussian_process"))
end

function minimal_failing_example()
    # some kind of kernel
    kernel = pykernels.ConstantKernel(constant_value = 1.0)

    # some data
    n = 20                                       # number of training points
    x = reshape(2.0 * π * rand(n), 1, n)         # predictors/features: 1 x n
    y = reshape(sin.(x) + 0.05 * randn(n)', 1, n) # predictands/targets: 1 x n

    # the model
    m = pyGP.GaussianProcessRegressor(kernel = kernel)

    # call fit
    m.fit(x, y)
    @info "fit successful"
end

init()
minimal_failing_example()
