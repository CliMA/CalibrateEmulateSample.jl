# src/show.jl
# All Base.show and Base.summary methods for CalibrateEmulateSample.jl.
# Included from CalibrateEmulateSample.jl after all submodule includes.

using .Utilities
using .Emulators
using .MarkovChainMonteCarlo
import .MarkovChainMonteCarlo: RWMetropolisHastings, pCNMetropolisHastings,
                                BarkerMetropolisHastings, AutodiffProtocol

# ── Utilities ─────────────────────────────────────────────────────────────────

# ElementwiseScaler

function Base.show(io::IO, es::ElementwiseScaler)
    print(io, "ElementwiseScaler: $(get_type(es))")
end

function Base.show(io::IO, ::MIME"text/plain", es::ElementwiseScaler)
    if get(io, :compact, false)
        show(io, es)
    else
        println(io, "ElementwiseScaler")
        println(io, "  scaling method : $(get_type(es))")
        print(io,   "  initialized    : $(!isempty(es.shift))")
    end
end

function Base.summary(io::IO, es::ElementwiseScaler)
    print(io, "ElementwiseScaler ($(get_type(es)))")
end

# Decorrelator

function Base.show(io::IO, dd::Decorrelator)
    out = "Decorrelator: decorrelate_with=$(get_decorrelate_with(dd))"
    if get_retain_var(dd) < 1.0
        out *= ", retain_var=$(get_retain_var(dd))"
    end
    print(io, out)
end

function Base.show(io::IO, ::MIME"text/plain", dd::Decorrelator)
    if get(io, :compact, false)
        show(io, dd)
    else
        println(io, "Decorrelator")
        println(io, "  decorrelate_with : $(get_decorrelate_with(dd))")
        if get_retain_var(dd) < 1.0
            println(io, "  retain_var       : $(get_retain_var(dd))")
        end
        print(io,   "  initialized      : $(!isempty(dd.data_mean))")
    end
end

function Base.summary(io::IO, dd::Decorrelator)
    print(io, "Decorrelator (decorrelate_with=$(get_decorrelate_with(dd)))")
end

# CanonicalCorrelation

function Base.show(io::IO, cc::CanonicalCorrelation)
    out = "CanonicalCorrelation:"
    if length(get_apply_to(cc)) > 0
        out *= " apply_to=$(get_apply_to(cc)[1])"
    end
    if get_retain_var(cc) < 1.0
        out *= " retain_var=$(get_retain_var(cc))"
    end
    print(io, out)
end

function Base.show(io::IO, ::MIME"text/plain", cc::CanonicalCorrelation)
    if get(io, :compact, false)
        show(io, cc)
    else
        println(io, "CanonicalCorrelation")
        if length(get_apply_to(cc)) > 0
            println(io, "  apply_to    : $(get_apply_to(cc)[1])")
        end
        if get_retain_var(cc) < 1.0
            println(io, "  retain_var  : $(get_retain_var(cc))")
        end
        print(io,   "  initialized : $(!isempty(cc.data_mean))")
    end
end

function Base.summary(io::IO, cc::CanonicalCorrelation)
    print(io, "CanonicalCorrelation (retain_var=$(get_retain_var(cc)))")
end

# LikelihoodInformed

function Base.show(io::IO, li::LikelihoodInformed)
    out = "LikelihoodInformed: iters=$(get_iters(li)), grad_type=$(get_grad_type(li))"
    if get_retain_info(li) < 1.0
        out *= ", retain_info=$(get_retain_info(li))"
    end
    print(io, out)
end

function Base.show(io::IO, ::MIME"text/plain", li::LikelihoodInformed)
    if get(io, :compact, false)
        show(io, li)
    else
        println(io, "LikelihoodInformed")
        println(io, "  iters       : $(get_iters(li))")
        println(io, "  grad_type   : $(get_grad_type(li))")
        if get_retain_info(li) < 1.0
            println(io, "  retain_info : $(get_retain_info(li))")
        end
        print(io,   "  initialized : $(!isempty(li.data_mean))")
    end
end

function Base.summary(io::IO, li::LikelihoodInformed)
    print(io, "LikelihoodInformed (iters=$(get_iters(li)), grad_type=$(get_grad_type(li)))")
end

# NoiseInjector

function Base.show(io::IO, ni::NoiseInjector)
    print(io, "NoiseInjector (use_noise=$(ni.use_noise), K=$(size(ni.K,1))×$(size(ni.K,2)))")
end

function Base.show(io::IO, ::MIME"text/plain", ni::NoiseInjector)
    if get(io, :compact, false)
        show(io, ni)
    else
        println(io, "NoiseInjector")
        println(io, "  use_noise        : $(ni.use_noise)")
        println(io, "  scaling          : $(ni.scaling)")
        println(io, "  K size           : $(size(ni.K,1))×$(size(ni.K,2))")
        print(io,   "  encoder_schedule : $(length(ni.encoder_schedule)) entries")
    end
end

function Base.summary(io::IO, ni::NoiseInjector)
    print(io, "NoiseInjector (use_noise=$(ni.use_noise), K=$(size(ni.K,1))×$(size(ni.K,2)))")
end

# ── Emulators ─────────────────────────────────────────────────────────────────

# Helper: print one encoder line — dimension reduction + processor chain — for
# a given space ("in" or "out"). Skips silently when no encoder acts on that space.
function _show_encoder_line(io::IO, enc_sch, raw_dim, space, label)
    enc_dim = get_encoded_dim(enc_sch, space)
    isnothing(enc_dim) && return
    print(io, "  ", label, ": ", raw_dim, " → ", enc_dim, "  ")
    first_name = true
    for (p, a) in enc_sch
        if a == space
            first_name || print(io, " → ")
            print(io, nameof(typeof(p)))
            first_name = false
        end
    end
    println(io)
end

# GaussianProcess

function Base.show(io::IO, x::GaussianProcess{P}) where {P}
    n = length(x.models)
    print(io, "GaussianProcess{", nameof(P), "} (", n, " model", n == 1 ? "" : "s", ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::GaussianProcess{P}) where {P}
    if get(io, :compact, false)
        show(io, x)
    else
        n = length(x.models)
        println(io, "GaussianProcess{", nameof(P), "}")
        println(io, "  n_models    : ", n, " (one per output dimension)")
        println(io, "  noise_learn : ", x.noise_learn)
        print(io,   "  pred_type   : ", nameof(typeof(x.prediction_type)))
    end
end

function Base.summary(io::IO, x::GaussianProcess{P}) where {P}
    n = length(x.models)
    print(io, "GaussianProcess{", nameof(P), "} (", n, " model", n == 1 ? "" : "s", ")")
end

# ScalarRandomFeatureInterface

function Base.show(io::IO, x::ScalarRandomFeatureInterface)
    print(io, "ScalarRandomFeatureInterface (", get_input_dim(x), "→1, ",
          get_n_features(x), " features, ", nameof(typeof(get_kernel_structure(x))), "(…))")
end

function Base.show(io::IO, ::MIME"text/plain", x::ScalarRandomFeatureInterface)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "ScalarRandomFeatureInterface")
        println(io, "  input_dim   : ", get_input_dim(x))
        println(io, "  n_features  : ", get_n_features(x))
        println(io, "  kernel      : ", nameof(typeof(get_kernel_structure(x))))
        println(io, "  decomp      : ", x.feature_decomposition)
        print(io,   "  built       : ", !isempty(x.rfms))
    end
end

function Base.summary(io::IO, x::ScalarRandomFeatureInterface)
    print(io, "ScalarRandomFeatureInterface (", get_input_dim(x), "→1, ",
          get_n_features(x), " features, ", nameof(typeof(get_kernel_structure(x))), "(…))")
end

# VectorRandomFeatureInterface

function Base.show(io::IO, x::VectorRandomFeatureInterface)
    print(io, "VectorRandomFeatureInterface (", get_input_dim(x), "→", get_output_dim(x), ", ",
          get_n_features(x), " features, ", nameof(typeof(get_kernel_structure(x))), "(…))")
end

function Base.show(io::IO, ::MIME"text/plain", x::VectorRandomFeatureInterface)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "VectorRandomFeatureInterface")
        println(io, "  input_dim   : ", get_input_dim(x))
        println(io, "  output_dim  : ", get_output_dim(x))
        println(io, "  n_features  : ", get_n_features(x))
        println(io, "  kernel      : ", nameof(typeof(get_kernel_structure(x))))
        println(io, "  decomp      : ", x.feature_decomposition)
        print(io,   "  built       : ", !isempty(x.rfms))
    end
end

function Base.summary(io::IO, x::VectorRandomFeatureInterface)
    print(io, "VectorRandomFeatureInterface (", get_input_dim(x), "→", get_output_dim(x), ", ",
          get_n_features(x), " features, ", nameof(typeof(get_kernel_structure(x))), "(…))")
end

# Emulator

function Base.show(io::IO, x::Emulator)
    mlt = get_machine_learning_tool(x)
    n_in, n_out = size(get_io_pairs(x), 1)
    print(io, "Emulator (", nameof(typeof(mlt)), ", ", n_in, "→", n_out, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::Emulator)
    if get(io, :compact, false)
        show(io, x)
    else
        mlt     = get_machine_learning_tool(x)
        n_in, n_out = size(get_io_pairs(x), 1)
        n_train = size(DataContainers.get_inputs(get_io_pairs(x)), 2)
        enc_sch = get_encoder_schedule(x)
        println(io, "Emulator")
        println(io, "  machine_learning_tool : ", nameof(typeof(mlt)))
        println(io, "  input_dim             : ", n_in)
        println(io, "  output_dim            : ", n_out)
        println(io, "  n_train               : ", n_train, " samples")
        _show_encoder_line(io, enc_sch, n_in,  "in",  "encoder (input)      ")
        _show_encoder_line(io, enc_sch, n_out, "out", "encoder (output)     ")
    end
end

function Base.summary(io::IO, x::Emulator)
    mlt = get_machine_learning_tool(x)
    n_in, n_out = size(get_io_pairs(x), 1)
    print(io, "Emulator (", nameof(typeof(mlt)), ", ", n_in, "→", n_out, ")")
end

# ForwardMapWrapper

function Base.show(io::IO, x::ForwardMapWrapper)
    n_in, n_out = size(get_io_pairs(x), 1)
    print(io, "ForwardMapWrapper (", n_in, "→", n_out, ", prior_dim=", ndims(get_prior(x)), ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::ForwardMapWrapper)
    if get(io, :compact, false)
        show(io, x)
    else
        n_in, n_out = size(get_io_pairs(x), 1)
        enc_sch     = get_encoder_schedule(x)
        ni          = x.noise_injector
        println(io, "ForwardMapWrapper")
        println(io, "  input_dim    : ", n_in)
        println(io, "  output_dim   : ", n_out)
        println(io, "  prior_dim    : ", ndims(get_prior(x)))
        _show_encoder_line(io, enc_sch, n_in,  "in",  "encoder (input)  ")
        _show_encoder_line(io, enc_sch, n_out, "out", "encoder (output) ")
        print(io,   "  noise_inject : ", !isnothing(ni) && ni.use_noise)
    end
end

function Base.summary(io::IO, x::ForwardMapWrapper)
    n_in, n_out = size(get_io_pairs(x), 1)
    print(io, "ForwardMapWrapper (", n_in, "→", n_out, ", prior_dim=", ndims(get_prior(x)), ")")
end

# ── MachineLearningTools ──────────────────────────────────────────────────────

# ── MarkovChainMonteCarlo ─────────────────────────────────────────────────────

# RWMetropolisHastings

function Base.show(io::IO, x::RWMetropolisHastings{PT, ADT}) where {PT, ADT <: AutodiffProtocol}
    print(io, "RWMetropolisHastings{$(nameof(ADT))}")
end

function Base.show(io::IO, ::MIME"text/plain", x::RWMetropolisHastings{PT, ADT}) where {PT, ADT <: AutodiffProtocol}
    if get(io, :compact, false)
        show(io, x)
    else
        print(io, "RWMetropolisHastings{$(nameof(ADT))}")
    end
end

function Base.summary(io::IO, x::RWMetropolisHastings{PT, ADT}) where {PT, ADT <: AutodiffProtocol}
    print(io, "RWMetropolisHastings{$(nameof(ADT))}")
end

# pCNMetropolisHastings

function Base.show(io::IO, x::pCNMetropolisHastings{D, T}) where {D, T <: AutodiffProtocol}
    print(io, "pCNMetropolisHastings{$(nameof(T))}")
end

function Base.show(io::IO, ::MIME"text/plain", x::pCNMetropolisHastings{D, T}) where {D, T <: AutodiffProtocol}
    if get(io, :compact, false)
        show(io, x)
    else
        print(io, "pCNMetropolisHastings{$(nameof(T))}")
    end
end

function Base.summary(io::IO, x::pCNMetropolisHastings{D, T}) where {D, T <: AutodiffProtocol}
    print(io, "pCNMetropolisHastings{$(nameof(T))}")
end

# BarkerMetropolisHastings

function Base.show(io::IO, x::BarkerMetropolisHastings{D, T}) where {D, T <: AutodiffProtocol}
    print(io, "BarkerMetropolisHastings{$(nameof(T))}")
end

function Base.show(io::IO, ::MIME"text/plain", x::BarkerMetropolisHastings{D, T}) where {D, T <: AutodiffProtocol}
    if get(io, :compact, false)
        show(io, x)
    else
        print(io, "BarkerMetropolisHastings{$(nameof(T))}")
    end
end

function Base.summary(io::IO, x::BarkerMetropolisHastings{D, T}) where {D, T <: AutodiffProtocol}
    print(io, "BarkerMetropolisHastings{$(nameof(T))}")
end

# MCMCWrapper

function Base.show(io::IO, mcmc::MCMCWrapper)
    n_par = ndims(mcmc.prior)
    sampler_name = nameof(typeof(mcmc.mh_proposal_sampler))
    print(io, "MCMCWrapper (", n_par, " param", n_par == 1 ? "" : "s", ", ", sampler_name, ")")
end

function Base.show(io::IO, ::MIME"text/plain", mcmc::MCMCWrapper)
    if get(io, :compact, false)
        show(io, mcmc)
    else
        n_par        = ndims(mcmc.prior)
        n_obs        = length(mcmc.observations)
        obs_dim      = isempty(mcmc.observations) ? nothing : length(first(mcmc.observations))
        sampler_name = nameof(typeof(mcmc.mh_proposal_sampler))
        enc_sch      = get_encoder_schedule(mcmc)
        println(io, "MCMCWrapper")
        println(io, "  prior_dim        : ", n_par, " parameter", n_par == 1 ? "" : "s")
        isnothing(obs_dim) || println(io, "  obs_dim          : ", obs_dim)
        println(io, "  n_obs            : ", n_obs, " sample", n_obs == 1 ? "" : "s")
        println(io, "  sampler          : ", sampler_name)
        _show_encoder_line(io, enc_sch, n_par,   "in",  "encoder (input)  ")
        isnothing(obs_dim) || _show_encoder_line(io, enc_sch, obs_dim, "out", "encoder (output)")
    end
end

function Base.summary(io::IO, mcmc::MCMCWrapper)
    n_par = ndims(mcmc.prior)
    sampler_name = nameof(typeof(mcmc.mh_proposal_sampler))
    print(io, "MCMCWrapper (", n_par, " param", n_par == 1 ? "" : "s", ", ", sampler_name, ")")
end
