using JLD2
using Statistics
using LinearAlgebra
using PyPlot
using CalibrateEmulateSample.UKI


function f(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]] 
end


function compute_foward(μ::Array{Float64,1}, Δt::Float64, nt::Int64)
    @assert(length(μ) == 6)
    nx = 3
    x0 = μ[1:nx]
    σ,r, β = μ[nx+1:6]
    
    xs = zeros(nx, nt+1)
    xs[:,1] = x0
    for i=2:nt+1
        xs[:,i] = xs[:,i-1] + Δt*f(xs[:,i-1], σ, r, β)
    end
    
    return xs
end



function compute_obs(xs::Array{Float64,2})
    # N_x × N_t
    x1, x2, x3 = xs[1,:]', xs[2,:]', xs[3,:]'
    obs = [x1; x2; x3; x1.^2; x2.^2; x3.^2]
end

# fix σ, learn r and β
# f = [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]]
# σ, r, β = 10.0, 28.0, 8.0/3.0
function run_Lorenz_ensemble(params_i, Tobs, Tspinup, Δt)
    T = Tobs + Tspinup
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    
    N_ens,  N_θ = size(params_i)
    
    
    g_ens = Vector{Float64}[]
    
    for i = 1:N_ens
        # σ, r, β = 10.0, 28.0, 8.0/3.0
        σ, r, β = params_i[i, :]
        σ, β = abs(σ), abs(β)
        
        
        μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    
        xs = compute_foward(μ, Δt, nt)
        obs = compute_obs(xs)
        # g: N_ens x N_data
        push!(g_ens, dropdims(mean(obs[:, nspinup : end], dims = 2), dims=2)) 
    end
    return hcat(g_ens...)'
end


function Data_Gen(T::Float64 = 360.0, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)
    
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    σ, r, β = 10.0, 28.0, 8.0/3.0
    
    μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    xs = compute_foward(μ, Δt, nt)
    
    obs = compute_obs(xs)
    
    ##############################################################################
    
    n_obs_box = Int64((T - Tspinup)/Tobs)
    obs_box = zeros(size(obs, 1), n_obs_box)
    
    n_obs = Int64(Tobs/Δt)
    
    for i = 1:n_obs_box
        n_obs_start = nspinup + (i - 1)*n_obs + 1
        obs_box[:, i] = mean(obs[:, n_obs_start : n_obs_start + n_obs - 1], dims = 2)
    end
    
    t_mean = vec(mean(obs_box, dims=2))
    #t_mean = vec(obs_box[:, 1])
    
    t_cov = zeros(Float64, size(obs, 1), size(obs, 1))
    for i = 1:n_obs_box
        t_cov += (obs_box[:,i] - t_mean) *(obs_box[:,i] - t_mean)'
    end
    
    @info obs_box
    t_cov ./= (n_obs_box - 1)
    
    @save "t_cov.jld2" t_cov
    @save "t_mean.jld2" t_mean
    return t_mean, t_cov
end



function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01, 
    N_iter::Int64 = 100)
    
    parameter_names = ["σ", "r", "β"]
    
    
    ens_func(θ_ens) = run_Lorenz_ensemble(θ_ens, Tobs, Tspinup, Δt)
    
    ukiobj = UKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov)
    
    
    for i in 1:N_iter
        # Note that the parameters are exp-transformed for use as input
        # to Cloudy
        params_i = deepcopy(ukiobj.θ_bar[end])
        
        @info "At iter ", i, " params_i : ", params_i
        
        update_ensemble!(ukiobj, ens_func) 
        
    end
    
    @info "θ is ", ukiobj.θ_bar[end], " θ_ref is ",  10.0, 28.0, 8.0/3.0

    return ukiobj
    
end


Tobs = 20.0
Tspinup = 30.0
T = Tspinup + 5*Tobs
Δt = 0.01

t_mean, t_cov = Data_Gen(T, Tobs, Tspinup, Δt)


# initial distribution is 
θ0_bar = [5.0 ; 5.0;  5.0]          # mean 
θθ0_cov = [0.5^2  0.0    0.0; 
          0.0    0.5^2  0.0;        # standard deviation
          0.0    0.0    0.5^2;]

N_ite = 20 

ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  Tobs, Tspinup, Δt, N_ite)
ites = Array(LinRange(1, N_ite+1, N_ite+1))
θ_bar = ukiobj.θ_bar
θθ_cov = ukiobj.θθ_cov

θ_bar_arr = abs.(hcat(θ_bar...))



N_θ = length(θ0_bar)
θθ_cov_arr = zeros(Float64, (N_θ, N_ite+1))
for i = 1:N_ite+1
    for j = 1:N_θ
        θθ_cov_arr[j, i] = sqrt(θθ_cov[i][j,j])
    end
end

@info "final result is ", θ_bar[end],  θθ_cov[end]

errorbar(ites, θ_bar_arr[1,:], yerr=3.0*θθ_cov_arr[1,:], fmt="--o",fillstyle="none", label="σ")
plot(ites, fill(10.0, N_ite+1), "--", color="gray")
errorbar(ites, θ_bar_arr[2,:], yerr=3.0*θθ_cov_arr[2,:], fmt="--o",fillstyle="none", label="r")
plot(ites, fill(28.0, N_ite+1), "--", color="gray")
errorbar(ites, θ_bar_arr[3,:], yerr=3.0*θθ_cov_arr[3,:], fmt="--o",fillstyle="none", label="β")
plot(ites, fill(8.0/3.0, N_ite+1), "--", color="gray")


xlabel("Iterations")
legend()
grid("on")
tight_layout()
savefig("Lorenz63-3para.png")
close("all")