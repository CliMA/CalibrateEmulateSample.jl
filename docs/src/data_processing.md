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

### Recommended: schedule for PCA dimension reduction

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
## Recommended: Get `encoder_kwargs` from `ObservationSeries` etc.

To transition more smoothly from the `EnsembleKalmanProcesses` infrastructure, one can also get the kwargs in the desired format from any [`Observation` and `ObservationSeries`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/observations/) objects, as well as [`ParameterDistribution`](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parameter_distributions/) objects used in the `EnsembleKalmanProcesses.jl` package.
```julia
# `prior::ParameterDistribution`
input_kwargs = get_kwargs_from(prior)

# `observation_series::ObservationSeries`
output_kwargs = get_kwargs_from(observation_series)

encoder_kwargs = merge(input_kwargs, output_kwargs)
```
!!! note "Observational noise notes"
    1. The `obs_noise_cov` object does not need to be a constructed `AbstractMatrix`, rather it can be any type of compactly stored matrix compatible with `EnsembleKalmanProcesses` (e.g., [here](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/observations/)).
    2. When `ObservationSeries` contains multiple observations, it is assumed that the covariance of the noise of all observations are the same. (If they are not, only the first will be taken by default in `get_kwargs_from(...)`
    

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

### Additional API
Once created, the user does not need to worry about encoding/decoding. However, we do provide an external API that can be used to encode both new data vectors (or matrix-of-columns), and structure matrices (i.e. covariances) for validation or error diagnosis.
```julia
# encoding new data: Vector, `DataContainer`, or Matrix viewed as columns
enc_in_data = encode_data(emulator, new_input_data, "in")
enc_out_data = encode_data(emulator, new_output_data, "out")

# decoding it
dec_in_data = decode_data(emulator, enc_in_data, "in")
dec_out_data = decode_data(emulator, enc_out_data, "out")

# new structure matrix: Matrix (typically acting as a covariance)
enc_in_cov = encode_structure_matrix(emulator, new_input_covariance, "in")
enc_out_cov = encode_structure_matrix(emulator, new_output_covariance, "out")

dec_in_cov = decode_structure_matrix(emulator, enc_in_cov, "in")
dec_out_cov = decode_structure_matrix(emulator, enc_out_cov, "out")
```


# Types of data processors

We currently provide two main types of data processing: the `DataContainerProcessor` and `PairedDataContainerProcessor`.

The `DataContainerProcessor` encodes "input" data agnostic of the "output" data, and vice versa, examples of current implementations are:
- `UnivariateAffineScaling`: such as `quartile_scale()`, `minmax_scale()`, and `zscore_scale()`, which apply some basic univariate scaling to the data in each dimension
- `Decorrelator`: such as `decorrelate_structure_mat()` and `decorrelate_sample_cov()`, or `decorrelate()` which perform [(truncated-)PCA](https://en.wikipedia.org/wiki/Singular_value_decomposition) using either the sample-estimated or user-provided covariance matrices (or their sum).

The `PairedDataContainerProcessor` encodes inputs (or outputs) using information of the both inputs and outputs in pairs
- `CanonicalCorrelation` - constructed with `canonical_correlation()`, which performs [canonical correlation analysis](https://en.wikipedia.org/wiki/Canonical_correlation) to process the pairs. In effect this performs PCA on the cross-correlation from input and output samples.
- `LikelihoodInformed` - constructed with `likelihood_informed()`, which uses the data or likelihood from the inverse problem at hand to build diagnostic matrices that are used to find informative directions for dimension reduction (e.g., [Cui, Zahm 2021](http://doi.org/10.1088/1361-6420/abeafb), [Baptista, Marzouk, Zahm 2022](https://arxiv.org/abs/2207.08670))

This is an extensible framework, and so new data processors can be added to this library.

!!! note "Some experiments"
    Some effects of the data processing (with older API) are outlined in a practical setting in the results and appendices of [Howland, Dunbar, Schneider, (2022)](https://doi.org/10.1029/2021MS002735).

## A note on `LinearMaps` and the internals for encoding large observations

To handle the heterogeneity and possible massive size of matrices involved with encoding observations, all `obs_noise_cov` and encoder-decoder objects are converted into linear maps, using [`LinearMaps.jl`](https://julialinearalgebra.github.io/LinearMaps.jl/stable/). For high-dimensional inputs or outputs (>3000), these are used in within iterative or (stochastically) approximate matrix-free methods, thus giving some possible additional error to gain better scaling in dimension. Such balances are controllable by the user with keywords that can be passed to certain encoders in the schedule.




