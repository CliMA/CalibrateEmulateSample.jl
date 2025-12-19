# [Data Processing and Dimension Reduction](@id data-proc)

## Overview 
When working with high-dimensional problems with modest training data pairs, the bottleneck of CES procedure is the training of a competent emulator. It is often necessary to process and dimensionally-reduce the data to ease the learning task of the emulator. To make results as transparent and reproducible as possible we provide a flexible (and extensible!) framework to create encoders and decoders for this purpose.

The framework works as follows
- An `encoder_schedule` defines the type of processing to be applied to input and/or output spaces
- The `encoder_schedule` is passed into the `Emulator` where it is initialized and stored. Sometimes it will require additional information, passed in as `encoder_kwargs`. 
- The encoder will be used automatically to encode training data, covariance matrices, and predictions within the Emulate and Sample routines.

An external API is also available using the `encode_data`, and `encode_structure_matrix` methods if needed.

## Defaults and recommendations
### Default schedule (no explicit dimension reduction)
When no schedule is provided, i.e. 
```julia
emulator = Emulator(machine_learning_tool, input_output_pairs)
```
The default schedule under-the-hood, is given by
```julia
schedule = (decorrelate_sample_cov(), "in_and_out")
```
If the user only provides the observational noise covariance, 
```julia
emulator = Emulator(
    machine_learning_tool,       
    input_output_pairs;
    encoder_kwargs = (; obs_noise_cov = obs_noise_cov),
)
```
The default schedule under-the-hood, is given by
```julia
schedule = [
    (decorrelate_sample_cov(), "in"),
    (decorrelate_structure_mat(), "out"), # uses obs_noise_cov
]
```

!!! note "Switch encoding off"
    To ensure that no encoding is happening, the user must pass in an empty schedule `encoder_schedule = []`

### Recommended schedule for PCA dimension reduction

Our recommendeded family to balance efficiency and reduce dimension is to use the `retain_var` kwargs.
```julia
retain_var_in = 0.99 # reduce dimension retaining 99% of input variance
retain_var_out = 0.95 # reduce dimension retaining 95% of output variance
encoder_schedule = [
    (decorrelate_sample_cov(retain_var = retain_var_in), "in"),
    (decorrelate_structure_mat(retain_var = retain_var_out), "out"),
]
encoder_kwargs = (; obs_noise_cov = obs_noise_cov)

emulator = Emulator(
    machine_learning_tool,       
    input_output_pairs;
    encoder_schedule = encoder_schedule,
    encoder_kwargs = encoder_kwargs,
)
```
## Getting `encoder_kwargs` from `EnsembleKalmanProcesses` objects

To transition more smoothly from the `EnsembleKalmanProcesses` infrastructure, one can also get the kwargs needed from [`Observation` and `ObservationSeries`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/observations/) objects, as well as [`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) objects used in the `EnsembleKalmanProcesses.jl` package.
```julia
# `prior::ParameterDistribution`
input_kwargs = get_kwargs_from(prior)

# `observation_series::ObservationSeries`
output_kwargs = get_kwargs_from(observation_series)

encoder_kwargs = merge(input_kwargs, output_kwargs)
```

## Building and interpreting encoder schedules

The user may provide an encoder schedule to transform data in a useful way. For example, mapping all output data each dimension to be bounded in [0,1]
```julia
simple_schedule = (minmax_scale(), "out")
```
The unit of the encoder contains a `DataProcessor`, the type of processing, and an string, whether to apply to input data,`"in"`, or output data `"out"`, or both `"in_and_out"`.

The encoder schedule can be a vector of several units that apply multiple `DataProcessors` in order:
```julia
complex_schedule = [
  (decorrelate_sample_cov(), "in"),
  (quartile_scale(), "in"), 
  (decorrelate_structure_mat(retain_var=0.95), "out"),
  (canonical_correlation(), "in_and_out"),
]
```
In this (rather unrealistic) chain;
1. The inputs are decorrelated with their sample mean and covariance (and projected to low dimensional subspace if necessary) i.e PCA
2. The scaled inputs are then subject to a "Robust" univariate scaling, mapping 1st-3rd quartiles to [0,1]
3. The outputs are decorrelated using an "output structure matrix" (provided to the emulator in the `encoder_kwargs` keyword parameter, e.g. as `(; obs_cov_noise =)`). Furthermore, apply a dimension-reduction to a space that retains 95% of the total variance. 
4. In the reduced input-output space, a canonical correlation analysis is performed. Data is oriented and reduced (if necessary) maximize the joint correlation between inputs and outputs.

!!! note "Default Encoder schedule"
    The current default encoder schedule applies `decorrelate_structure_mat()` if a structure matrix (input or output) is provided, else it applies `decorrelate_sample_cov()`.



## Creating an emulator with a schedule

The schedule is then passed into the Emulator, along with the data and desired structure matrices
```julia
emulator = Emulator(
    machine_learning_tool,       
    input_output_pairs;
    encoder_schedule = complex_schedule,
    encoder_kwargs = (; obs_noise_cov = obs_noise_cov),
)
```
Note that due to the item `(decorrelate_structure_mat(retain_var=0.95), "out")` in the schedule, we must provide an output structure matrix. In this case, we provide `obs_noise_cov`.

# Types of data processors

We currently provide two main types of data processing: the `DataContainerProcessor` and `PairedDataContainerProcessor`.

The `DataContainerProcessor` encodes "input" data agnostic of the "output" data, and vice versa, examples of current implementations are:
- `UnivariateAffineScaling`: such as `quartile_scale()`, `minmax_scale()`, and `zscore_scale()`, which apply some basic univariate scaling to the data in each dimension
- `Decorrelator`: such as `decorrelate_structure_mat()` and `decorrelate_sample_cov()`, or `decorrelate()` which perform [(truncated-)PCA](https://en.wikipedia.org/wiki/Singular_value_decomposition) using either the sample-estimated or user-provided covariance matrices (or their sum).

The `PairedDataContainerProcessor` encodes inputs (or outputs) using information of the both inputs and outputs in pairs
- `CanonicalCorrelation` - constructed with `canonical_correlation()`, which performs [canonical correlation analysis](https://en.wikipedia.org/wiki/Canonical_correlation) to process the pairs. In effect this performs PCA on the cross-correlation from input and output samples.
- [Coming soon] `LikelihoodInformed` - this will use the data or likelihood from the inverse problem at hand to build diagnostic matrices that are used to find informative directions for dimension reduction. In particular we build generalizations of current frameworks (e.g., [Cui, Zahm 2021](http://doi.org/10.1088/1361-6420/abeafb), [Baptista, Marzouk, Zahm 2022](https://arxiv.org/abs/2207.08670)) to use the latest EKP iterations.

This is an extensible framework, and so new data processors can be added to this library.

!!! note "Some experiments"
    Some effects of the data processing (with older API) are outlined in a practical setting in the results and appendices of [Howland, Dunbar, Schneider, (2022)](https://doi.org/10.1029/2021MS002735).
