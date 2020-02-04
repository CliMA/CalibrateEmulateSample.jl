"""
    eks_iter(prob::CESProblem, θs, fθs)

Perform an iteration of the ensemble Kalman sampler (EKS), returning a new set of proposed `θs`.

See eqs. (5.4a-b) from https://arxiv.org/abs/1903.08866.
"""
function eks_iter(prob::CESProblem, θs::Vector{Vector{FT}}, fθs::Vector{Vector{FT}}) where {FT}
    covθ = cov(θs)
    meanθ = mean(θs)

    m = mean(fθs)

    J = length(θs)
    CG = [dot(fθk - m, fθj - prob.obs, prob.space)/J for fθj in fθs, fθk in fθs]

    Δt = FT(1) / (norm(CG) + sqrt(eps(FT)))

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

