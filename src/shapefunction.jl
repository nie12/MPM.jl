#################
# Interpolation #
#################

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

@inline update_particle_domain!(pt::MaterialPoint, ::Type{Tent}) = nothing


# uGIMP, cpGIMP
# -------------

abstract type GIMP <: Interpolation end
struct uGIMP  <: GIMP end
struct cpGIMP <: GIMP end

@inline update_particle_domain!(pt::MaterialPoint, ::Type{uGIMP}) = nothing
@generated function update_particle_domain!(pt::MaterialPoint{dim}, ::Type{cpGIMP}) where {dim}
    # Charlton, T. J., W. M. Coombs, and C. E. Augarde.
    # "iGIMP: an implicit generalised interpolation material point method for large deformations."
    # Computers & Structures 190 (2017): 108-125.
    # Eq. (39)
    return quote
        @_inline_meta
        U² = pt.F' ⋅ pt.F
        @inbounds pt.lp = Vec(@ntuple $dim i -> pt.lp₀[i] * √U²[i,i])
         # @inbounds pt.lp = Vec(@ntuple $dim i -> pt.lp₀[i] * pt.F[i,i]) # this is original
    end
end

@inline function value(::Type{<: GIMP}, xv::Real, L::Real, xp::Real, lp::Real)
    d = abs(xp - xv)
    return d < lp     ? 1 - (d^2 + lp^2) / (2L*lp) :
           d < L - lp ? 1 - d / L                  :
           d < L + lp ? (L + lp - d)^2 / (4L*lp)   : zero(d)
end
@inline function derivative(::Type{<: GIMP}, xv::Real, L::Real, xp::Real, lp::Real)
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
        @inbounds @ntuple $dim i -> value(interp, N.x[i], N.L[i], pt.x[i], pt.lp[i])
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
        @inbounds @ntuple $dim i -> derivative(interp, ∇N.body.x[i], ∇N.body.L[i], pt.x[i], pt.lp[i])
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
