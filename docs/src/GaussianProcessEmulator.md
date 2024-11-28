# Gaussian Process Emulator

One type of `MachineLearningTool` we provide for emulation is a Gaussian process.
Gaussian processes are a generalization of the Gaussian probability distribution, extended to functions rather than random variables.
They can be used for statistical emulation, as they provide both mean and covariances.
To build a Gaussian process, we first define a prior over all possible functions, by choosing the covariance function or kernel.
The kernel describes how similar two outputs `(y_i, y_j)` are, given the similarities between their input values `(x_i, x_j)`.
Kernels encode the functional form of these relationships and are defined by hyperparameters, which are usually initially unknown to the user.
To learn the posterior Gaussian process, we condition on data using Bayes theorem and optimize the hyperparameters of the kernel. 
Then, we can make predictions to predict a mean function and covariance for new data points.

A useful resource to learn about Gaussian processes is [Rasmussen and Williams (2006)](http://gaussianprocess.org/gpml/).


# User Interface

`CalibrateEmulateSample.jl` allows the Gaussian process emulator to be built using
either [`GaussianProcesses.jl`](https://stor-i.github.io/GaussianProcesses.jl/latest/) 
or [`ScikitLearn.jl`](https://scikitlearnjl.readthedocs.io/en/latest/models/#scikitlearn-models). Different packages may be optimized for different settings, we recommend users give both a try, and checkout the individual package documentation to make a choice for their problem setting. 

To use `GaussianProcesses.jl`, define the package type as
```julia
gppackage = Emulators.GPJL()
```

To use `ScikitLearn.jl`, define the package type as
```julia
gppackage = Emulators.SKLJL()
```

Initialize a basic Gaussian Process with
```julia
gauss_proc = GaussianProcess(gppackage)
```

This initializes the prior Gaussian process. 
We train the Gaussian process by feeding the `gauss_proc` alongside the data into the `Emulator` struct and optimizing the hyperparameters,
described [here](https://clima.github.io/CalibrateEmulateSample.jl/dev/emulate/#Typical-construction-from-Lorenz_example.jl).

# Prediction Type

You can specify the type of prediction when initializing the Gaussian Process emulator.
The default type of prediction is to predict data, `YType()`. 
You can create a latent function type prediction with

```julia
gauss_proc = GaussianProcess(
    gppackage,
    prediction_type = FType())

```


# Kernels

The Gaussian process above assumes the default kernel: the Squared Exponential kernel, also called 
the [Radial Basis Function (RBF)](https://en.wikipedia.org/wiki/Radial_basis_function_kernel). 
A different type of kernel can be specified when the Gaussian process is initialized. 
Read more about [kernel options here](https://www.cs.toronto.edu/~duvenaud/cookbook/).


## GPJL 
For the `GaussianProcess.jl` package, there are [a range of kernels to choose from](https://stor-i.github.io/GaussianProcesses.jl/latest/kernels). 
For example, 
```julia
using GaussianProcesses
my_kernel = GaussianProcesses.Mat32Iso(0., 0.)      # Create a Matern 3/2 kernel with lengthscale=0 and sd=0
gauss_proc = GaussianProcess(
    gppackage;
    kernel = my_kernel )
```
You do not need to provide useful hyperparameter values when you define the kernel, as these are learned in 
`optimize_hyperparameters!(emulator)`.

You can also combine kernels together through linear operations, for example,
```julia
using GaussianProcesses
kernel_1 = GaussianProcesses.Mat32Iso(0., 0.)      # Create a Matern 3/2 kernel with lengthscale=0 and sd=0
kernel_2 = GaussianProcesses.Lin(0.)               # Create a linear kernel with lengthscale=0
my_kernel = kernel_1 + kernel_2                    # Create a new additive kernel
gauss_proc = GaussianProcess(
    gppackage;
    kernel = my_kernel )
```

## SKLJL
Alternatively if you are using the `ScikitLearn.jl` package, you can [find the list of kernels here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process). 
These need this preamble:
```julia
using PyCall
using ScikitLearn
const pykernels = PyNULL()
function __init__()
    copy!(pykernels, pyimport("sklearn.gaussian_process.kernels"))
end
```
Then they are accessible, for example, as
```julia
my_kernel = pykernels.RBF(length_scale = 1)
gauss_proc = GaussianProcess(
    gppackage;
    kernel = my_kernel )
```
You can also combine multiple ScikitLearn kernels via linear operations in the same way as above.

# Learning additional white noise

Often it is useful to learn the discrepancy between the Gaussian process prediction and the data, by learning additional white noise. Though one often knows, and provides, the discrepancy between the true model and data with an observational noise covariance; the additional white kernel can help account for approximation error from the selected Gaussian process kernel and the true model. This is added with the Boolean keyword `noise_learn` when initializing the Gaussian process. The default is true. 

```julia
gauss_proc = GaussianProcess(
    gppackage;
    noise_learn = true )
```

When `noise_learn` is true, an additional white noise kernel is added to the kernel. This white noise is present
across all parameter values, including the training data. 
The scale parameters of the white noise kernel are learned in `optimize_hyperparameters!(emulator)`. 

You may not need to learn the noise if you already have a good estimate of the noise from your training data, and if the Gaussian process kernel is well specified. 
When `noise_learn` is false, a small additional regularization is added for stability.
The default value is `1e-3` but this can be chosen through the optional argument `alg_reg_noise`:

```julia
gauss_proc = GaussianProcess(
    gppackage;
    noise_learn = false,
    alg_reg_noise = 1e-3 )
```

