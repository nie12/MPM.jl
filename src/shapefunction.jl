#################
# Interpolation #
#################

abstract type Interpolation end

struct Tent <: Interpolation end

@inline neighbor_element(::Type{Tent}, i::Int) = i:i
@inline update_particle_domain!(pt::MaterialPoint, ::Type{Tent}) = nothing

abstract type GIMP <: Interpolation end
struct uGIMP  <: GIMP end
struct cpGIMP <: GIMP end

@inline neighbor_element(::Type{<: GIMP}, i::Int) = i-1:i+1
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


#############
# LineShape #
#############

@enum Dirichlet begin
    FREE = 100
    FIXED = 0
    RFIXED = -1
    LFIXED =  1
end

@inline Base.:*(x::Dirichlet, y::Dirichlet) = abs(Int(x)) < abs(Int(y)) ? x : y

mutable struct LineShape{interp <: Interpolation, T}
    x::T
    L::T
    dirichlet::Dirichlet
end
@inline function LineShape{interp}(x::Real, y::Real) where {interp}
    T = promote_type(typeof(x), typeof(y))
    LineShape{interp, T}(x, y, FREE)
end

@inline function derivative(shape::LineShape, xp::Real, lp::Real)
    dxp = TensorArrays.dualize(xp)
    TensorArrays.extract_gradient(value(shape, dxp, lp), xp)
    # gradient(xp -> value(shape, xp, lp), xp) # This simple code is really slow when `LineShape` is mutable struct
end

@inline function value(shape::LineShape{Tent}, xp::Real, lp::Real)
    xv = shape.x
    L = shape.L
    d = abs(xp - xv)
    d < L ? 1 - d/L : zero(d)
end

@inline function value(shape::LineShape{<: GIMP}, xp::Real, lp::Real)
    xv = shape.x
    L = shape.L
    bounds = (xp - lp - xv, xp + lp - xv)
    lhs = clamp.(bounds, -L, 0)
    rhs = clamp.(bounds,  0, L)
    A = @inbounds begin
        (2 + (lhs[1] + lhs[2])/L) * (lhs[2]-lhs[1]) / 2 +
        (2 - (rhs[1] + rhs[2])/L) * (rhs[2]-rhs[1]) / 2
    end
    return A / 2lp
end


#################
# ShapeFunction #
#################

struct ShapeFunction{interp <: Interpolation, dim, T} <: AbstractVector{LineShape{interp, dim}}
    shapes::NTuple{dim, LineShape{interp, T}}
end

@inline Base.size(N::ShapeFunction) = length(N.shapes)
@inline @propagate_inbounds Base.getindex(N::ShapeFunction, i::Int) = N.shapes[i]

@inline (N::ShapeFunction{interp, dim})(pt::MaterialPoint{dim}) where {interp, dim} = prod(apply(N, pt))

@generated function apply(N::ShapeFunction{interp, dim}, pt::MaterialPoint{dim}) where {interp, dim}
    return quote
        @_inline_meta
        @inbounds @ntuple $dim i -> value(N[i], pt.x[i], pt.lp[i])
    end
end

@generated function setdirichlet!(N::ShapeFunction{interp, dim}, dirichlets::NTuple{dim, Dirichlet}) where {interp, dim}
    return quote
        @_inline_meta
        @inbounds @nexprs $dim d -> N.shapes[d].dirichlet = dirichlets[d]
        nothing
    end
end

@generated function getdirichlet(N::ShapeFunction{interp, dim}) where {interp, dim}
    return quote
        @_inline_meta
        @inbounds @ntuple $dim d -> N.shapes[d].dirichlet
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
        @inbounds @ntuple $dim i -> derivative(∇N.body[i], pt.x[i], pt.lp[i])
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
