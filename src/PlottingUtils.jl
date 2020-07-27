module PlottingUtils

using Plots
using JLD
using LinearAlgebra
using Statistics
using BoundingSphere

export eki_sphere_evol
export plot_outputs
export plot_eki_params

function eki_sphere_evol(eki_u)
    #N-dimensional measures
    eki_center = zeros((length(eki_u), length(eki_u[1][1,:])))
    eki_radius = zeros(length(eki_u))
    eki_center_jump = zeros(length(eki_u))
    #Iterate through EKI stages
    for i in 1:length(eki_u)
        eki_center[i,:], eki_radius[i] = boundingsphere(
            mapslices(x->[x], eki_u[i], dims=2)[:])
        if i > 1
            eki_center_jump[i] = sqrt( (eki_center[i,:]-eki_center[i-1,:])'*(
                eki_center[i,:]-eki_center[i-1,:]) )
        end
    end
    return eki_radius, eki_center_jump
end

function plot_eki_params(eki_u)
    #One-dimensional measures
    eki_mean = zeros( (length(eki_u), length(eki_u[1][1,:])) )
    eki_min =  zeros( (length(eki_u), length(eki_u[1][1,:])) )
    eki_max =  zeros( (length(eki_u), length(eki_u[1][1,:])) )
    #Iterate through EKI stages
    for i in 1:length(eki_u)
        eki_mean[i,:] = mean(eki_u[i], dims=1)
        eki_min[i,:] = minimum(eki_u[i], dims=1)
        eki_max[i,:] = maximum(eki_u[i], dims=1)
    end

    for j in 1:length(eki_u[1][1,:])
        p = plot((eki_u[1][:,j]), seriestype = :scatter)
        for i in 2:length(eki_u)
            plot!(p, eki_u[i][:,j], seriestype = :scatter)
        end
        savefig(string("EKI_points_param_",j,".png"))

        plot(eki_mean[:,j], ribbon=(
            -eki_min[:,j].+eki_mean[:,j], eki_max[:,j].-eki_mean[:,j]),
            label=string("parameter ", j), lw=2)
        xlabel!("EKI iteration")
        savefig(string("evol_param_",j,".png"))
    end
    return
end

function plot_outputs(eki_g, les_g, num_var, num_heights)
    for sim in 1:length(num_var)
        h_r = range(0, 1, length=Integer(num_heights[sim]))
        for k in 1:num_var[sim]
            inf_lim = Integer(
                num_heights[sim]*(k-1) + (num_heights[1:sim-1]'*num_var[1:sim-1]))+1
            sup_lim = Integer( inf_lim + num_heights[sim] - 1)
            y_model_mean = mean(eki_g[end][:,inf_lim:sup_lim],dims=1)[1,:]

            plot(les_g[inf_lim:sup_lim] , h_r, label="LES", lw=2)
            plot!(y_model_mean, h_r, label="SCM", lw=2, ls=:dash)
            ylabel!("Normalized height")
            savefig(string("sim_",sim,"_output_variable",k,".png"))
        end
    end
    return
end

end #module