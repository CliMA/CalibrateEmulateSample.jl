module GModel

export run_G
export run_G_ensemble
export lorenz_forward

include("GModel_common.jl")

function run_ensembles(settings, lorenz_params, nd, N_ens)
    nthreads = Threads.nthreads()
    g_ens = zeros(nthreads, nd, N_ens)
    Threads.@threads for i in 1:N_ens
        tid = Threads.threadid()
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[tid, :, i] = lorenz_forward(settings, lorenz_params[i])
    end
    g_ens = dropdims(sum(g_ens, dims = 1), dims = 1)
    return g_ens
end

end # module
