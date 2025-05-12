abstract type ForwardMapType end

## G*exp(X)
struct LinearExp{AM <: AbstractMatrix} <: ForwardMapType
    input_dim::Int
    output_dim::Int
    G::AM
end

# columns of X are samples
function forward_map(X::AVorM, model::LE) where {LE <: LinearExp, AVorM <: AbstractVecOrMat}
    return model.G * exp.(X)
end

# columns of X are samples
function jac_forward_map(X::AM, model::LE) where {AM <: AbstractMatrix, LE <: LinearExp}
    # dGi / dXj = G_ij exp(x_j) = G.*exp.(mat with repeated x_j rows)
    #    return [G * exp.(Diagonal(r)) for r in eachrow(X')] # correct but extra multiplies
    return [model.G .* exp.(reshape(c, 1, :)) for c in eachcol(X)]
end

function jac_forward_map(X::AV, model::LE) where {AV <: AbstractVector, LE <: LinearExp}
    return jac_forward_map(reshape(X, :, 1), model)
end
