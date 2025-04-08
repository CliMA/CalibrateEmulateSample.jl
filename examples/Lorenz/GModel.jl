module GModel

export run_G
export run_G_ensemble
export lorenz_forward

using ChunkSplitter

include("GModel_common.jl")

function run_ensembles(settings, lorenz_params, nd, N_ens)
    nthreads = Threads.nthreads()

    chunked_ensemble = chunks(1:N_ens, n = nthreads) # could probs do without this package. But gives (tid, idx for tid)

    g_ens = zeros(nd, N_ens)
    Threads.@threads for e_idx in chunked_ensemble
        for i in e_idx
            # run the model with the current parameters, i.e., map θ to G(θ)
            g_ens[:, i] = lorenz_forward(settings, lorenz_params[i])
        end
    end
    return g_ens
end

end # module
