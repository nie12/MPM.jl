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
@inline function update_particle_domain!(pt::MaterialPoint{dim, T}, ::Type{cpGIMP}) where {dim, T}
    # Charlton, T. J., W. M. Coombs, and C. E. Augarde.
    # "iGIMP: an implicit generalised interpolation material point method for large deformations."
    # Computers & Structures 190 (2017): 108-125.
    # Eq. (39)
    U² = pt.F' ⋅ pt.F
    pt.lp = Vec{dim, T}(i -> pt.lp₀[i] * √U²[i,i])
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

# Directly call AD methods because using anonymous function is really slow when `LineShape` is mutable struct
@inline function derivative(shape::LineShape, xp::Real, lp::Real)
    dxp = TensorArrays.dualize(xp)
    TensorArrays.extract_gradient(value(shape, dxp, lp), xp)
end
@inline function derivative(shape::LineShape, xp::Real, lp::Real, ::Symbol)
    dxp = TensorArrays.dualize(xp)
    dN = value(shape, dxp, lp)
    tuple(TensorArrays.extract_gradient(dN, xp), TensorArrays.extract_value(dN, xp))
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
    plhs = (xp - lp) - xv
    prhs = (xp + lp) - xv
    # skip the particle beyond the boundary
    (shape.dirichlet == LFIXED && plhs < -L) && return zero(plhs)
    (shape.dirichlet == RFIXED && prhs >  L) && return zero(prhs)
    # computed domains of left and right hand sides
    lhs = clamp.((plhs, prhs), -L, 0)
    rhs = clamp.((plhs, prhs),  0, L)
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

@generated function (N::ShapeFunction{interp, dim})(pt::MaterialPoint{dim}) where {interp, dim}
    return quote
        @_inline_meta
        @inbounds prod(@ntuple $dim i -> value(N[i], pt.x[i], pt.lp[i]))
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

@generated function (∇N::GradientShapeFunction{interp, dim})(pt::MaterialPoint{dim}, ::Symbol) where {interp, dim}
    exps = map(1:dim) do d
        Expr(:call, :*, [i == d ? :(∇Nᵢ[$i]) : :(Nᵢ[$i]) for i in 1:dim]...)
    end
    return quote
        @_inline_meta
        @inbounds begin
            @nexprs $dim d -> res_d = derivative(∇N.body[d], pt.x[d], pt.lp[d], :all)
            ∇Nᵢ = @ntuple $dim d -> res_d[1]
            Nᵢ = @ntuple $dim d -> res_d[2]
            return tuple(Vec($(exps...)), prod(Nᵢ))
        end
    end
end
@inline function (∇N::GradientShapeFunction)(pt::MaterialPoint)
    @inbounds ∇N(pt, :all)[1]
end
