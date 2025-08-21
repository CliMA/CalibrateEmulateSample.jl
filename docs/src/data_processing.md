# [Data Processing and Dimension Reduction](@id data-proc)

## Overview 
When working with high-dimensional problems with modest training data pairs, the bottleneck of CES procedure is the training of a competent emulator. It is often necessary to process and dimensionally-reduce the data to ease the learning task of the emulator. We provide a flexible (and extensible!) framework to create encoders and decoders for this purpose. The framework works as follows
- An `encoder_schedule` defines the type of processing to be applied to input and/or output spaces
- The `encoder_schedule` is passed into the `Emulator` where it is initialized and stored. 
- The encoder will be used automatically to encode training data and predictions within the Emulate and Sample routines.

An external API is also available using the `encode_data`, and `encode_structure_matrix` methods if needed.

## Define an encoder schedule

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

!!! note "Switch encoding off"
   To ensure that no encoding is happening, the user must pass in an empty schedule `encoder_schedule = []`


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
- [Coming soon] `LikelihoodInformed` - this will use the data or likelihood from the inverse problem at hand to build diagnostic matrices that are used to find informative directions for dimension reduction (e.g., [Cui, Zahm 2021](http://doi.org/10.1088/1361-6420/abeafb), [Baptista, Marzouk, Zahm 2022](https://arxiv.org/abs/2207.08670))

This is an extensible framework, and so new data processors can be added to this library.

!!! note "Some experiments"
    Some effects of the data processing (with older API) are outlined in a practical setting in the results and appendices of [Howland, Dunbar, Schneider, (2022)](https://doi.org/10.1029/2021MS002735).
