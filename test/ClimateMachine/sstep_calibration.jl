using Distributions
using JLD
using ArgParse
# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities


function reconstruct_ek_obj(iteration_)
    # Construct EK object
    versions = readlines("versions_$(iteration_).txt")
    open("$(versions[1]).output/$(versions[1])", "r") do io
        n_params = length([parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0])
        u_names = [split(line, "(")[1] for (index, line) in enumerate(eachline(io)) if index%2 == 0]
    end
    u = zeros(length(versions), n_params)
    for (ens_index, version_) in enumerate(versions)
        open("$(version_).output/$(version_)", "r") do io
            u[ens_index, :] = [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0]
        end
    end
    open("recovered_params_$(iteration_).txt", "w") do io
        write(io, u)
    end
end

s = ArgParseSettings()
@add_arg_table s begin
    "--iteration"
        help = "Calibration iteration number"
        arg_type = Int
        default = 1
end
parsed_args = parse_args(ARGS, s)
iteration_ = parsed_args["iteration"]
reconstruct_ek_obj(iteration_)