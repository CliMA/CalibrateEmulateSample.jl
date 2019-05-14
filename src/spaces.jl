import LinearAlgebra: norm, dot

"""
    HilbertSpace

A `Space <: HilbertSpace` defines methods for
- `norm(x, ::Space)`
- `dot(x, y, ::Space)`
"""
abstract type HilbertSpace
end

"""
    DefaultSpace

Uses the default `norm` and `dot` implementations (e.g. `Array`s and `Array`s of `Array`s).
"""
struct DefaultSpace <: HilbertSpace
end
norm(x, ::DefaultSpace) = norm(x)
dot(x,y, ::DefaultSpace) = dot(x,y)


struct CovarianceSpace{M} <: HilbertSpace
    Γ::M
end
norm(x, c::CovarianceSpace) = x'*(c.Γ\x)
dot(x,y, c::CovarianceSpace) = x'*(c.Γ\y)

# TODO: CovarianceSpace/PDSpace with a positive definite matrix
