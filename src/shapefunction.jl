abstract type Interpolation end

# Tent
# ----

struct Tent <: Interpolation end

@inline function value(::Type{Tent}, xv::Real, L::Real, xp::Real, lp::Real)
    d = abs(xp - xv)
    d < L ? 1 - d/L : zero(d)
end
@inline function derivative(::Type{Tent}, xv::Real, L::Real, xp::Real, lp::Real)
    d = abs(xp - xv)
    sgn = sign(xp - xv)
    d < L ? -sgn / L : zero(d)
end


# uGIMP
# -----

struct uGIMP <: Interpolation end

@inline function value(::Type{uGIMP}, xv::Real, L::Real, xp::Real, lp::Real)
    d = abs(xp - xv)
    return d < lp     ? 1 - (d^2 + lp^2) / (2L*lp) :
           d < L - lp ? 1 - d / L                  :
           d < L + lp ? (L + lp - d)^2 / (4L*lp)   : zero(d)
end
@inline function derivative(::Type{uGIMP}, xv::Real, L::Real, xp::Real, lp::Real)
    d = abs(xp - xv)
    sgn = sign(xp - xv)
    return d < lp     ? -sgn * d / (L*lp)            :
           d < L - lp ? -sgn / L                     :
           d < L + lp ? -sgn * (L+ lp - d) / (2L*lp) : zero(d)
end

#################
# ShapeFunction #
#################

struct ShapeFunction{interp <: Interpolation, dim, T}
    x::Vec{dim, T}
    L::Vec{dim, T}
end
@inline ShapeFunction{interp}(x::Vec{dim, T}, L::Vec{dim, T}) where {interp, dim, T} = ShapeFunction{interp, dim, T}(x, L)

@inline (N::ShapeFunction{interp, dim})(pt::MaterialPoint{dim}) where {interp, dim} = prod(apply(N, pt))

@generated function apply(N::ShapeFunction{interp, dim}, pt::MaterialPoint{dim}) where {interp, dim}
    return quote
        @_inline_meta
        @inbounds @ntuple $dim i -> value(interp, N.x[i], N.L[i], pt.x[i], pt.lₚ[i])
    end
end

#########################
# GradientShapeFunction #
#########################

struct GradientShapeFunction{interp, dim, T}
    body::ShapeFunction{interp, dim, T}
end
@inline LinearAlgebra.adjoint(N::ShapeFunction) = GradientShapeFunction(N)

@generated function apply(∇N::GradientShapeFunction{interp, dim}, pt::MaterialPoint{dim}) where {interp, dim}
    return quote
        @_inline_meta
        @inbounds @ntuple $dim i -> derivative(interp, ∇N.body.x[i], ∇N.body.L[i], pt.x[i], pt.lₚ[i])
    end
end

@generated function (∇N::GradientShapeFunction{interp, dim})(pt::MaterialPoint{dim}) where {interp, dim}
    exps = map(1:dim) do d
        Expr(:call, :*, [i == d ? :(∇Nᵢ[$i]) : :(Nᵢ[$i]) for i in 1:dim]...)
    end
    return quote
        @_inline_meta
        Nᵢ = apply(∇N.body, pt)
        ∇Nᵢ = apply(∇N, pt)
        @inbounds Vec($(exps...))
    end
end
