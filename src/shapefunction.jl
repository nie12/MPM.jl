"""
    ShapeFunction1D(x::Real, l::Real)

`ShapeFunction1D` is defined at 1D global coordinate `x` with nodal spacing `l`.
"""
struct ShapeFunction1D{T <: Real}
    xi::T
    l::T
end

@inline function value(N::ShapeFunction1D{T}, x::Real) where {T}
    d = abs(x - N.xi)
    return d ≤ N.l ? (1 - d / N.l) : zero(d)
end

@inline function TensorArrays.gradient(N::ShapeFunction1D{T}, x::Real) where {T}
    gradient(x -> value(N, x), x)
end


struct ShapeFunction{dim, T <: Real}
    shapes::NTuple{dim, ShapeFunction1D{T}}
end

@inline function ShapeFunction(coord::Vec{dim, T}, l::NTuple{dim, T}) where {dim, T}
    shapes = ntuple(Val(dim)) do i
        @inbounds return ShapeFunction1D(coord[i], l[i])
    end
    return ShapeFunction(shapes)
end

@generated function value(N::ShapeFunction{dim}, x::Vec{dim}) where {dim}
    return quote
        @_inline_meta
        @inbounds return prod(@ntuple $dim i -> value(N.shapes[i], x[i]))
    end
end
@inline (N::ShapeFunction{dim})(x::Vec{dim}) where {dim} = value(N, x)

@inline function TensorArrays.gradient(N::ShapeFunction{dim}, x::Vec{dim}) where {dim}
    gradient(x -> value(N, x), x)
end


struct GradientShapeFunction{dim, T <: Real}
    N::ShapeFunction{dim, T}
end

@inline LinearAlgebra.adjoint(N::ShapeFunction) = GradientShapeFunction(N)

@inline (∇N::GradientShapeFunction{dim})(x::Vec{dim}) where {dim} = gradient(∇N.N, x)

for ShapeFunctionType in (ShapeFunction, GradientShapeFunction)
    @eval begin
        @inline (N::$ShapeFunctionType{dim})(x::NTuple{dim, Real}) where {dim} = N(Vec(x))
        @inline (N::$ShapeFunctionType{dim})(x::Vararg{Real, dim}) where {dim} = N(Vec(x))
        @inline (N::$ShapeFunctionType{dim})(p::Particle{dim}) where {dim} = N(p.x)
    end
end
