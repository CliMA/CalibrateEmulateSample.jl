module Netcdf_utils

using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using Glob

export obs_LES
export get_profile
export get_clima_profile

# Read truth data from PyCLES
function obs_LES(y_names::Array{String, 1},
                    sim_dir::String,
                    ti::Float64,
                    tf::Float64;
                    z_scm::Union{Array{Float64, 1}, Nothing} = nothing,
                    ) where {FT<:AbstractFloat}
    
    y_names_les = get_les_names(y_names, sim_dir)
    y_highres = get_profile(sim_dir, y_names_les, ti = ti, tf = tf)
    y_tvar = get_timevar_profile(sim_dir, y_names_les,
        ti = ti, tf = tf, z_scm=z_scm)
    if !isnothing(z_scm)
        y_ = zeros(0)
        z_les = get_profile(sim_dir, ["z_half"])
        num_outputs = Integer(length(y_highres)/length(z_les))
        for i in 1:num_outputs
            y_itp = interpolate( (z_les,), 
                y_highres[1 + length(z_les)*(i-1) : i*length(z_les)],
                Gridded(Linear()) )
            append!(y_, y_itp(z_scm))
        end
    else
        y_ = y_highres
    end
    return y_, y_tvar
end

function get_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf::Float64=0.0,
                     getFullHeights=false)

    if length(var_name) == 1 && occursin("z_half", var_name[1])
        prof_vec = nc_fetch(sim_dir, "profiles", var_name[1])
    else
        t = nc_fetch(sim_dir, "timeseries", "t")
        dt = t[2]-t[1]
        ti_diff, ti_index = findmin( broadcast(abs, t.-ti) )
        tf_diff, tf_index = findmin( broadcast(abs, t.-tf) )
        
        prof_vec = zeros(0)
        # If simulation does not contain values for ti or tf, return high value
        if ti_diff > dt || tf_diff > dt
            for i in 1:length(var_name)
                var_ = nc_fetch(sim_dir, "profiles", var_name[i])
                append!(prof_vec, 1.0e4*ones(length(var_[:, 1])))
            end
        else
            for i in 1:length(var_name)
                var_ = nc_fetch(sim_dir, "profiles", var_name[i])
                # LES vertical fluxes are per volume, not mass
                if occursin("resolved_z_flux", var_name[i])
                    rho_half=nc_fetch(sim_dir, "reference", "rho0_half")
                    var_ = var_.*rho_half
                end
                append!(prof_vec, mean(var_[:, ti_index:tf_index], dims=2))
            end
        end
    end
    return prof_vec 
end

function get_clima_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf::Float64=0.0,
                     z_les::Union{Array{Float64, 1}, Nothing} = nothing)

    ds_name = glob("$(sim_dir)/*.nc")[1]
    ds = NCDataset(ds_name)

    if length(var_name) == 1 && occursin("z", var_name[1])
        prof_vec = Array(ds[var_name[1]])
    else
        # Time in Datetime format
        t = Array(ds["time"])
        # Convert to float (seconds)
        t = [(time_ .- t[1]).value/1000.0 for time_ in t]
        dt = t[2]-t[1]
        ti_diff, ti_index = findmin( broadcast(abs, t.-ti) )
        tf_diff, tf_index = findmin( broadcast(abs, t.-tf) )
        
        prof_vec = zeros(0)
        # If simulation does not contain values for ti or tf, return high value
        if ti_diff > dt || tf_diff > dt
            for i in 1:length(var_name)
                var_ = Array(ds[var_name[i]])
                append!(prof_vec, 1.0e4*ones(length(var_[:, 1])))
            end
        else
            for i in 1:length(var_name)
                var_ = Array(ds[var_name[i]])
                if !isnothing(z_les)
                    z_ = Array(ds["z"])
                    var_itp = interpolate( (z_, 1:tf_index-ti_index+1),
                        var_[:, 1:tf_index-ti_index+1],
                        ( Gridded(Linear()), NoInterp() ))
                    var_ = var_itp(z_les, 1:tf_index-ti_index+1), dims=1)
                end
                append!(prof_vec, mean(var_[:, ti_index:tf_index], dims=2))
            end
        end
    end
    close(ds)
    return prof_vec 
end

function get_timevar_profile(sim_dir::String,
                     var_name::Array{String,1};
                     ti::Float64=0.0,
                     tf::Float64=0.0,
                     getFullHeights=false,
                     z_scm::Union{Array{Float64, 1}, Nothing} = nothing)

    t = nc_fetch(sim_dir, "timeseries", "t")
    dt = t[2]-t[1]
    ti_diff, ti_index = findmin( broadcast(abs, t.-ti) )
    tf_diff, tf_index = findmin( broadcast(abs, t.-tf) )
    prof_vec = zeros(0, length(ti_index:tf_index))

    # If simulation does not contain values for ti or tf, return high value
    for i in 1:length(var_name)
        var_ = nc_fetch(sim_dir, "profiles", var_name[i])
        # LES vertical fluxes are per volume, not mass
        if occursin("resolved_z_flux", var_name[i])
            rho_half=nc_fetch(sim_dir, "reference", "rho0_half")
            var_ = var_.*rho_half
        end
        prof_vec = cat(prof_vec, var_[:, ti_index:tf_index], dims=1)
    end
    if !isnothing(z_scm)
        if !getFullHeights
            z_les = get_profile(sim_dir, ["z_half"])
        else
            z_les = get_profile(sim_dir, ["z"])
        end
        num_outputs = Integer(length(prof_vec[:, 1])/length(z_les))
        prof_vec_zscm = zeros(0, length(ti_index:tf_index))
        maxvar_vec = zeros(num_outputs) 
        for i in 1:num_outputs
            maxvar_vec[i] = maximum(var(prof_vec[1 + length(z_les)*(i-1) : i*length(z_les), :], dims=2))
            prof_vec_itp = interpolate( (z_les, 1:tf_index-ti_index+1),
                prof_vec[1 + length(z_les)*(i-1) : i*length(z_les), :],
                ( Gridded(Linear()), NoInterp() ))
            prof_vec_zscm = cat(prof_vec_zscm,
                prof_vec_itp(z_scm, 1:tf_index-ti_index+1), dims=1)
        end
        # Get correlation matrix, substitute NaNs by zeros
        cov_mat = cov(prof_vec_zscm, dims=2)
        for i in 1:num_outputs
            cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), 1 + length(z_scm)*(i-1) : i*length(z_scm)] = (
               cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), 1 + length(z_scm)*(i-1) : i*length(z_scm)]
               - Diagonal(cov_mat[1 + length(z_scm)*(i-1) : i*length(z_scm), 1 + length(z_scm)*(i-1) : i*length(z_scm)])
               + maxvar_vec[i]*Diagonal(ones(length(z_scm), length(z_scm))))
        end
    else
        cov_mat = cov(prof_vec, dims=2)
    end
    return cov_mat
end

function get_les_names(scm_y_names::Array{String,1}, sim_dir::String)
    y_names = deepcopy(scm_y_names)
    if "thetal_mean" in y_names
        if occursin("GABLS",sim_dir) || occursin("Soares",sim_dir)
            y_names[findall(x->x=="thetal_mean", y_names)] .= "theta_mean"
        else
            y_names[findall(x->x=="thetal_mean", y_names)] .= "thetali_mean"
        end
    end
    if "total_flux_qt" in y_names
        y_names[findall(x->x=="total_flux_qt", y_names)] .= "resolved_z_flux_qt"
    end
    if "total_flux_h" in y_names && (occursin("GABLS",sim_dir) || occursin("Soares",sim_dir))
        y_names[findall(x->x=="total_flux_h", y_names)] .= "resolved_z_flux_theta"
    elseif "total_flux_h" in y_names
        y_names[findall(x->x=="total_flux_h", y_names)] .= "resolved_z_flux_thetali"
    end
    if "u_mean" in y_names
        y_names[findall(x->x=="u_mean", y_names)] .= "u_translational_mean"
    end
    if "v_mean" in y_names
        y_names[findall(x->x=="v_mean", y_names)] .= "v_translational_mean"
    end
    if "tke_mean" in y_names
        y_names[findall(x->x=="tke_mean", y_names)] .= "tke_nd_mean"
    end
    return y_names
end

function nc_fetch(dir, nc_group, var_name)
    find_prev_to_name(x) = occursin("Output", x)
    split_dir = split(dir, ".")
    sim_name = split_dir[findall(find_prev_to_name, split_dir)[1]+1]
    ds = NCDataset(string(dir, "/stats/Stats.", sim_name, ".nc"))
    ds_group = ds.group[nc_group]
    ds_var = deepcopy( Array(ds_group[var_name]) )
    close(ds)
    return Array(ds_var)
end

end #module