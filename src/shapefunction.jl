abstract type AbstractInterpolation end

#################
# ShapeFunction #
#################

struct ShapeFunction{Interpolation <: AbstractInterpolation, dim, T}
    interpolation::Interpolation
    x::Vec{dim, T}
    l::Vec{dim, T}
end

#########################
# GradientShapeFunction #
#########################

struct GradientShapeFunction{Interpolation <: AbstractInterpolation, dim, T}
    N::ShapeFunction{Interpolation, dim, T}
end
@inline LinearAlgebra.adjoint(N::ShapeFunction) = GradientShapeFunction(N)
@inline (∇N::GradientShapeFunction)(pt::MaterialPoint) = gradient(∇N.N, pt)


# LinearInterpolation
# -------------------

struct LinearInterpolation <: AbstractInterpolation end

@generated function value(N::ShapeFunction{LinearInterpolation, dim}, x::Vec{dim}) where {dim}
    return quote
        @_inline_meta
        @inbounds prod(@ntuple $dim i -> begin
                           l = N.l[i]
                           d = abs(x[i] - N.x[i])
                           d < l ? 1 - d / l : zero(d)
                       end)
    end
end
@inline (N::ShapeFunction{LinearInterpolation})(pt::MaterialPoint) = value(N, pt.x)

@inline function TensorArrays.gradient(N::ShapeFunction{LinearInterpolation}, x::Vec)
    gradient(x -> value(N, x), x)
end
@inline function TensorArrays.gradient(N::ShapeFunction{LinearInterpolation}, pt::MaterialPoint)
    gradient(N, pt.x)
end


# GeneralizedInterpolation
# ------------------------

struct GeneralizedInterpolation <: AbstractInterpolation end

@generated function value(N::ShapeFunction{GeneralizedInterpolation, dim}, x::Vec{dim}, lₚ_::Vec{dim}) where {dim}
    return quote
        @_inline_meta
        @inbounds prod(@ntuple $dim i -> begin
                           l = N.l[i]
                           lₚ = lₚ_[i]
                           d = abs(x[i] - N.x[i])
                           d < lₚ     ? 1 - (d^2+ lₚ^2) / (2l*lₚ) :
                           d < l - lₚ ? 1 - d / l                 :
                           d < l + lₚ ? (l + lₚ - d)^2 / (4l*lₚ)  : zero(d)
                       end)
    end
end
@inline (N::ShapeFunction{GeneralizedInterpolation})(pt::MaterialPoint) = value(N, pt.x, pt.lₚ)

@inline function TensorArrays.gradient(N::ShapeFunction{GeneralizedInterpolation}, x::Vec, lₚ::Vec)
    gradient(x -> value(N, x, lₚ), x)
end
@inline function TensorArrays.gradient(N::ShapeFunction{GeneralizedInterpolation}, pt::MaterialPoint)
    gradient(N, pt.x, pt.lₚ)
end
