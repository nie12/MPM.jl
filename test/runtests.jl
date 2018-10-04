using LinearAlgebra
using MPM, TensorArrays
using Test

function gausspoints(::Type{T}, ::Val{1}) where {T <: Real}
    v = T(sqrt(1/3))
    return (Vec{1}((-v,)),
            Vec{1}(( v,)))
end
function gausspoints(::Type{T}, ::Val{2}) where {T <: Real}
    v = T(sqrt(1/3))
    return (Vec{2}((-v, -v)),
            Vec{2}(( v, -v)),
            Vec{2}(( v,  v)),
            Vec{2}((-v,  v)))
end
function gausspoints(::Type{T}, ::Val{3}) where {T <: Real}
    v = T(sqrt(1/3))
    return (Vec{3}((-v, -v, -v)),
            Vec{3}(( v, -v, -v)),
            Vec{3}(( v,  v, -v)),
            Vec{3}((-v,  v, -v)),
            Vec{3}((-v, -v,  v)),
            Vec{3}(( v, -v,  v)),
            Vec{3}(( v,  v,  v)),
            Vec{3}((-v,  v,  v)))
end

include("materialpoint.jl")
include("shapefunction.jl")
include("node.jl")
include("grid.jl")
include("problem.jl")
