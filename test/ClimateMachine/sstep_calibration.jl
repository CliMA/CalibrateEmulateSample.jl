using Distributions
using JLD
using ArgParse
using NCDatasets
# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Netcdf_utils

function reconstruct_ek_obj(iteration_)
    # Construct EK object
    versions = readlines("versions_$(iteration_).txt")
    n_params = 0
    u_names = []
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
    println(u_names)
    println(u)
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

# Get observations
yt = zeros(0)
yt_var_list = []
y_names = Array{String, 1}[]
push!(y_names, ["u_mean", "v_mean"])
les_dir = string("/groups/esm/ilopezgo/Output.", "GABLS",".iles128wCov")
sim_dir = string("Output.", "GABLS",".00000")
z_scm = get_clima_profile(sim_dir, ["z"])
yt_, yt_var_ = obs_LES(y_names[2], les_dir, ti[2], tf[2], z_scm = z_scm)
append!(yt, yt_)
push!(yt_var_list, yt_var_)