"""
    neki_iter(prob::SolusProblem, θs, fθs)

Peform an iteration of NEKI, returning a new set of proposed `θs`.

See eqs. (5.4a-b) from https://arxiv.org/abs/1903.08866.
"""
function neki_iter(prob::SolusProblem, θs, fθs)
    covθ = cov(θs)
    meanθ = mean(θs)
    
    m = mean(fθs)

    J = length(θs)
    CG = [dot(fθk - m, fθj - prob.obs, prob.space)/J for fθj in fθs, fθk in fθs]

    Δt = 1.0 / (norm(CG) + 1e-8)
    
    implicit = lu( I + Δt .* (covθ * inv(cov(prob.prior))) ) # todo: incorporate means
    Z = covθ * (cov(prob.prior) \ mean(prob.prior)) 
    
    noise = MvNormal(covθ)
    
    # θs .- Δt .* (CG*(θs .- Ref(meanθ))) .+ Δt .* Ref(covθ * (cov(prob.prior) \ mean(prob.prior)))
    
    # compute next set of θs
    map(enumerate(θs)) do (j, θj)
        X = sum(enumerate(θs)) do (k, θk)
            CG[j,k]*(θk-meanθ)
        end
        rhs = θj .- Δt .* X .+ Δt .* Z
        (implicit \ rhs) .+ sqrt(2*Δt)*rand(noise)
    end
end
    
