mutable struct Node{dim, T, interp} <: AbstractVector{T}
    N::ShapeFunction{interp, dim, T}
    m::T
    mv::Vec{dim, T}
    f::Vec{dim, T}
end
@inline function Node(N::ShapeFunction{interp, dim, T}) where {interp, dim, T}
    m = zero(T)
    mv = zero(Vec{dim, T})
    f = zero(Vec{dim, T})
    return Node(N, m, mv, f)
end

@inline Base.size(::Node{dim}) where {dim} = (dim,)
@inline @propagate_inbounds Base.getindex(node::Node, i::Int) = node.N.shapes[i].x

@inline function reset!(node::Node{dim, T}) where {dim, T}
    node.m = zero(T)
    node.mv = zero(Vec{dim, T})
    node.f = zero(Vec{dim, T})
    return node
end
