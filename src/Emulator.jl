module Emulator


abstract type EmulatorType end

# We will define the different emulator types after the general statements

"""
    Emulator

Structure used to represent a general emulator
"""

struct Emulator
    "training inputs and outputs, data stored in columns"
    input_output_pairs::PairedDataContainer{FT}
    "Emulator type"
    emulator_type::EmulatorType
    "mean of input; 1 × input_dim"
    input_mean::Array{FT}
    "square root of the inverse of the input covariance matrix; input_dim × input_dim"
    sqrt_inv_input_cov::Union{Nothing, Array{FT, 2}}
    "whether to fit models on normalized inputs ((inputs - input_mean) * sqrt_inv_input_cov)"
    normalize_inputs::Bool
    "whether to fit models on normalized outputs outputs / standardize_outputs_factor"
    standardize_outputs::Bool
    
    
end

"""
    Emulator(input_output_pairs, emulator_type; normalized::Bool = true)

Constructor for the Emulator
"""
function Emulator(input_output_pairs,
                  emulator_type;
                  normalize_inputs::Bool = true,
                  standardize_outputs::Bool = false,
                  standardize_outputs_factor::Union{Array{FT,1}, Nothing}=nothing)

    # For Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)

    # Normalize the inputs? 
    input_mean = reshape(mean(get_inputs(input_output_pairs), dims=2), :, 1) #column vector
    sqrt_inv_input_cov = nothing
    if normalize_inputs
        # Normalize (NB the inputs have to be of) size [input_dim × N_samples] to pass to GPE())
        sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(input_output_pairs), dims=2))))
        GPinputs = sqrt_inv_input_cov * (get_inputs(input_output_pairs).-input_mean)
    else
        GPinputs = get_inputs(input_output_pairs)
    end

    # Standardize the outputs?
    if standardize_outputs
	output_values = get_outputs(input_output_pairs) ./ standardize_outputs_factor
	obs_noise_cov = obs_noise_cov ./ (standardize_outputs_factor .* standardize_outputs_factor')
    else
	output_values = get_outputs(input_output_pairs)
    end

    return Emulator(input_output_pairs,
                    emulator_type,
                    input_mean,
                    sqrt_inv_input_cov,
                    normalize_inputs,
                    standardize_outputs,
                    standardize_outputs_factor)
end    


function optimize_emulator_parameters()

end

function predict()

end









end
