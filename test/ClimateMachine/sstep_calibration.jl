using Distributions
using JLD
using ArgParse
using NCDatasets
using LinearAlgebra
# Import Calibrate-Emulate-Sample modules
using CalibrateEmulateSample.Priors
using CalibrateEmulateSample.EKP
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Netcdf_utils

function ek_update(iteration_)
    # Construct EK object
    versions = readlines("versions_$(iteration_).txt")
    n_params = 0
    u_names = String[]
    open("$(versions[1]).output/$(versions[1])", "r") do io
        append!(u_names, [strip(string(split(line, "(")[1]), [' ']) for (index, line) in enumerate(eachline(io)) if index%3 == 2])
        n_params = length(u_names)
    end
    u = zeros(length(versions), n_params)
    for (ens_index, version_) in enumerate(versions)
        open("$(version_).output/$(version_)", "r") do io
            u[ens_index, :] = [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0]
        end
    end

    # Get observations
    yt = zeros(0)
    yt_var_list = []
    y_names = ["u_mean", "v_mean"]
    les_dir = string("/groups/esm/ilopezgo/Output.", "GABLS",".iles128wCov")
    z_scm = get_clima_profile("$(versions[1]).output", ["z"])
    println(size(z_scm))
    println(z_scm)
    yt_, yt_var_ = obs_LES(y_names, les_dir, 0.0, 360.0)
    yt_var_ = yt_var_ + Matrix(0.1I, size(yt_var_)[1], size(yt_var_)[2])
    append!(yt, yt_)
    push!(yt_var_list, yt_var_)

    # Get outputs
    g_names = ["u", "v"]
    g_ = get_clima_profile("$(versions[1]).output", g_names, ti=0.0, tf=360.0, z_les = Array(range(3.125, 400, length=128)))
    g_ens = zeros(length(versions), length(g_))
    for (ens_index, version) in enumerate(versions)
        g_ens[ens_index, :] = get_clima_profile("$(version).output", g_names, ti=0.0, tf=360.0, z_les = Array(range(3.125, 400, length=128)))
    end
    # Construct EKP
    ekobj = EKP.EKObj(u, u_names, yt_, yt_var_, Inversion())
    # Advance EKP
    EKP.update_ensemble!(ekobj, g_ens)
    # Get step
    ek_final_result = ekobj.u[end]
    return ek_final_result
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

