mutable struct SnapShot{dim, T, N, M, Ms} <: AbstractArray{MaterialPoint{dim, T, M, Ms}, N}
    steps::NTuple{dim, T}
    limits::NTuple{dim, Tuple{T, T}}
    t::T
    points::Array{MaterialPoint{dim, T, M, Ms}, N}
end

Base.size(s::SnapShot) = size(s.points)

@inline function Base.getindex(s::SnapShot{dim, T, N}, cartesian::Vararg{Int, N}) where {dim, T, N}
    @boundscheck checkbounds(s, cartesian...)
    @inbounds s.points[cartesian...]
end

@inline function Base.setindex!(s::SnapShot{dim, T, N}, pt::MaterialPoint{dim}, cartesian::Vararg{Int, N}) where {dim, T, N}
    @boundscheck checkbounds(s, cartesian...)
    @inbounds s.points[cartesian...] = pt
end

function Base.similar(s::SnapShot{dim, <: Real}, ::Type{MaterialPoint{dim, T, M, Ms}}, dims::Dims{N}) where {dim, T, N, M, Ms}
    points = Array{MaterialPoint{dim, T, M, Ms}}(undef, dims)
    for i in eachindex(points)
        points[i] = MaterialPoint{dim, T, M, Ms}()
    end
    return SnapShot{dim, T, N, M, Ms}(s.steps, s.limits, s.t, points)
end
