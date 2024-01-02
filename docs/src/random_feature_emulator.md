# Random Feature Emulator

!!! note "Have a go with Gaussian processes first"
    We recommend that users first try `GaussianProcess` for their problems. As random features are a more recent tool, the training procedures and interfaces are still experimental and in development. 

Random features provide a flexible framework to approximates a Gaussian process. Using random sampling of features, the method is a low-rank approximation leading to advantageous scaling properties (with the number of training points, input, and output dimensions). In the infinite sample limit, there are often (known) explicit Gaussian process kernels that the random feature representation converges to.

We provide two types of `MachineLearningTool` for random feature emulation, the `ScalarRandomFeatureInterface` and the `VectorRandomFeatureInterface`.

The `ScalarRandomFeatureInterface` closely mimics the role of a `GaussianProcess` package, by training a scalar-output function distribution. It can be applied to multidimensional output problems (as with `GaussianProcess`) by relying on data processing tools, such as performed when the `decorrelate=true` keyword argument is provided to the `Emulator`.

The `VectorRandomFeatureInterface`, when applied to multidimensional problems, directly trains a function distribution between multi-dimensional spaces. This approach is not restricted to the data processing of the scalar method (though this can still be helpful). It can be cheaper to evaluate, but on the other hand the training can be more challenging/computationally expensive.

Building a random feature interface is similar to building a Gaussian process: one defines a kernel to encode similarities between outputs ``(y_i,y_j)`` based on inputs ``(x_i,x_j)``. Additionally, one must specify the number of random feature samples to be taken to build the emulator.

# Recommended configuration

Below is listed a recommended configuration that is flexible and requires learning relatively few parameters. Users can increase `r` to balance flexibility against having more kernel hyperparameters to learn.

```julia
using CalibrateEmulateSample.Emulators
# given input_dim, output_dim, and a PairedDataContainer

# define number of features for prediction
n_features = 400 # number of features for prediction 

# define kernel 
nugget = 1e8*eps() # small nugget term
r = 1 # start with smallest rank 
lr_perturbation = LowRankFactor(r, nugget) 
nonsep_lrp_kernel = NonseparableKernel(lr_perturbation) 

# configure optimizer
optimizer_options = Dict(
    "verbose" => true, # print diagnostics for optimizer
    "n_features_opt" => 100, # use less features during hyperparameter optimization/kernel learning
    "cov_sample_multiplier" => 1.0, # use to reduce/increase number of samples in initial cov estimation stage
)

machine_learning_tool = VectorRandomFeatureInterface(n_features, input_dim, output_dim, optimizer_options = optimizer_options)
```
Users can change the kernel complexity with `r`, and the number of features for prediciton with `n_features` and optimization with `n_features_opt`. 

# User Interface

`CalibrateEmulateSample.jl` allows the random feature emulator to be built using the external package [`RandomFeatures.jl`](https://github.com/CliMA/RandomFeatures.jl). In the notation of this package's documentation, our interface allows for families of `RandomFourierFeature` objects to be constructed with different Gaussian distributions of the "`xi`" a.k.a weight distribution, and with a learnable "`sigma`", a.k.a scaling parameter.

!!! note "Relating features and kernels"
    The parallels of random features and gaussian processes can be quite strong. For example:
    - The restriction to `RandomFourierFeature` objects is a restriction to the approximation of shift-invariant kernels (i.e. ``K(x,y) = K(x-y)``)
    - The restriction of the weight ("`xi`") distribution to Gaussians is a restriction of approximating squared-exponential kernels. Other distributions (e.g. student-t) leads to other kernels (e.g. Matern)
    
The interfaces are defined minimally with

```julia 
srfi = ScalarRandomFeatureInterface(n_features, input_dim; ...)
vrfi = VectorRandomFeatureInterface(n_features, input_dim, output_dim; ...)
```

This will build an interface around a random feature family based on `n_features` features and mapping between spaces of dimenstion `input_dim` to `1` (scalar), or `output_dim` (vector).

## The `kernel_structure` keyword - for flexibility

To adjust the expressivity of the random feature family one can define the keyword argument `kernel_structure`. The more expressive the kernel, the more hyperparameters are learnt in the optimization.  

We have two types,
```julia
separable_kernel = SeparableKernel(input_cov_structure, output_cov_structure)
nonseparable_kernel = NonseparableKernel(cov_structure)
```
where the `cov_structure` implies some imposed user structure on the covariance structure. The basic covariance structures are given by 
```julia
1d_cov_structure = OneDimFactor() # the problem dimension is 1
diagonal_structure = DiagonalFactor() # impose diagonal structure (e.g. ARD kernel)
cholesky_structure = CholeskyFactor() # general positive definite matrix
lr_perturbation = LowRankFactor(r) # assume structure is a rank-r perturbation from identity
```
All covariance structures (except `OneDimFactor`) have their final positional argument being a "nugget" term adding ``+\epsilon I`` to the covariance structure. Set to 1 by default.

The current default kernels are as follows:
```julia
scalar_default_kernel = SeparableKernel(LowRankFactor(Int(ceil(sqrt(input_dim)))), OneDimFactor())
vector_default_kernel = SeparableKernel(LowRankFactor(Int(ceil(sqrt(output_dim)))), LowRankFactor(Int(ceil(sqrt(output_dim)))))
```
!!! note "Relating covariance structure and training"
    The parallels between random feature and Gaussian process also extends to the hyperparameter learning. For example,
    - A `ScalarRandomFeatureInterface` with a `DiagonalFactor` input covariance structure approximates a Gaussian process with automatic relevance determination (ARD) kernel, where one learns a lengthscale in each dimension of the input space
    
## The `optimizer_options` keyword - for performance

Passed as a dictionary, this keyword allows the user to configure many options from their defaults in the hyperparameter optimization. The optimizer itself relies on the [`EnsembleKalmanProcesses`](https://github.com/CliMA/EnsembleKalmanProcesses.jl) package.

We recommend users experiment with a subset of these flags. At first enable
```julia
Dict("verbose" => true)
```
If the covariance sampling takes too long, run with multithreading (e.g. `julia --project -t n_threads script.jl`). Sampling is embarassingly parallel so this acheives near linear scaling,

If sampling still takes too long, try setting
```julia
Dict(
    "cov_sample_multiplier" => csm,
    "train_fraction" => tf,
)
```
- Decreasing `csm` (default `10.0`) towards `0.0` directly reduces the number of samples to estimate a covariance matrix in the optimizer, by using a shrinkage estimator - the more shrinkage the more approximation (suggestion, keep shrinkage amount below `0.2`).
- Increasing `tf` towards `1` changes the train-validate split, reducing samples but increasing cost-per-sample and reducing the available validation data (default `0.8`, suggested range `(0.5,0.95)`).

If optimizer convergence stagnates or is too slow, or if it terminates before producing good results, try:
```julia
Dict(
    "n_ensemble" => n_e, 
    "n_iteration" => n_i,
    "localization" => loc,
    "scheduler" => sch,
)
```
We suggest looking at the [`EnsembleKalmanProcesses`](https://github.com/CliMA/EnsembleKalmanProcesses.jl) documentation for more details; but to summarize
- Reducing optimizer samples `n_e` and iterations `n_i` reduces computation time.
- If `n_e` becomes less than the number of hyperparameters, the updates will fail and a localizer must be specified in `loc`.
- If the algorithm terminates at `T=1` and resulting emulators looks unacceptable one can change or add arguments in `sch` e.g. `DataMisfitController("on_terminate"=continue)`

!!! note
    Widely robust defaults here are a work in progress

## Key methods

To interact with the kernel/covariance structures we have standard `get_*` methods along with some useful functions
- `cov_structure_from_string(string,dim)` creates a basic covariance structure from a predefined string: `onedim`, `diagonal`, `cholesky`, `lowrank` etc. and a dimension
- `calculate_n_hyperparameters(in_dim, out_dim, kernel_structure)` calculates the number of hyperparameters created by using the given kernel structure (can be applied to the covariance structure individually too)
- `build_default_priors(in_dim, out_dim, kernel_structure)` creates a `ParameterDistribution` for the hyperparameters based on the kernel structure. This serves as the initialization of the training procedure.

## Example families and their hyperparameters

### Scalar: ``\mathbb{R}^5 \to \mathbb{R}`` at defaults
```julia
using CalibrateEmulateSample.Emulators
input_dim = 5
# build the default scalar kernel directly (here it will be a rank-3 perturbation from the identity)
scalar_default_kernel = SeparableKernel(
    cov_structure_from_string("lowrank", input_dim),
    cov_structure_from_string("onedim", 1)
) 

calculate_n_hyperparameters(input_dim, scalar_default_kernel) 
# answer = 19, 18 for the covariance structure, and one scaling parameter

build_default_prior(input_dim, scalar_default_kernel)
# builds a 3-entry distribution
# 3-dim positive distribution 'input_lowrank_diagonal'
# 15-dim unbounded distribution 'input_lowrank_U'
# 1-dim positive distribution `sigma`
```
### Vector, separable: ``\mathbb{R}^{25} \to \mathbb{R}^{50}`` at defaults
Or take a diagonalized 8-dimensional input, and assume full 6-dimensional output

```julia
using CalibrateEmulateSample.Emulators
input_dim = 25
output_dim = 50
# build the default vector kernel directly (here it will be a rank-5 input and rank-8 output)
vector_default_kernel = SeparableKernel(
    cov_structure_from_string("lowrank", input_dim),
    cov_structure_from_string("lowrank", output_dim)
)

calculate_n_hyperparameters(input_dim, output_dim, vector_default_kernel) 
# answer = 539; 130 for input, 408 for the output, and 1 scaling

build_default_prior(input_dim, output_dim, vector_default_kernel)
# builds a 5-entry distribution
# 5-dim positive distribution 'input_lowrank_diagonal'
# 125-dim unbounded distribution 'input_lowrank_U'
# 8-dim positive distribution 'output_lowrank_diagonal'
# 400-dim unbounded distribution 'output_lowrank_U'
# 1-dim postive distribution `sigma`
```

### Vector, nonseparable: ``\mathbb{R}^{25} \to \mathbb{R}^{50}`` 
The following represents the most general kernel case.

!!! note "Use low-rank/diagonls representations where possible"
    The following is far too general, leading to large numbers of hyperparameters
```julia
using CalibrateEmulateSample.Emulators
input_dim = 25
output_dim = 50
eps = 1e-8
# build a full-rank nonseparable vector kernel
vector_general_kernel = NonseparableKernel(CholeskyFactor(eps))

calculate_n_hyperparameters(input_dim, output_dim, vector_general_kernel)
# answer = 781876; 781875 for the joint input-output space, and 1 scaling

build_default_prior(input_dim, output_dim, vector_default_kernel)
# builds a 2-entry distribution
# 781875-dim unbounded distribution 'full_cholesky'
# 1-dim positive distribution `sigma`
```

See the API for more details.


