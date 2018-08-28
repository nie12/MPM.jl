# 1D shape function defined in the global coordinate system
# `l` must be unique in the axis for now.
# Should define `l₊` and `l₋` instead of `l` to represent nonlinear spaces?
struct ShapeFunction{T <: Real}
    xi::T
    l::T
    function ShapeFunction{T}(xi::Real, l::Real) where {T}
        new{T}(xi, l)
    end
end
@inline ShapeFunction(xi::Real, l::Real) = ShapeFunction{promote_type(typeof(xi), typeof(l))}(xi, l)

@inline Base.step(shape::ShapeFunction) = shape.l

@inline function value(shape::ShapeFunction{T}, x::Real) where {T}
    d = norm(x - shape.xi)
    return d ≤ shape.l ? (1 - d / shape.l) : zero(T)
end

@inline function TensorArrays.gradient(shape::ShapeFunction, x::Real)
    gradient(@inline(function(x) value(shape, x) end), x)
end
