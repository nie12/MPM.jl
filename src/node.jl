mutable struct Node{dim, T, interp} <: AbstractVector{T}
    N::ShapeFunction{interp, dim, T}
    m::T
    v::Vec{dim, T}
    f::Vec{dim, T}
end
@inline function Node(N::ShapeFunction{interp, dim, T}) where {interp, dim, T}
    m = zero(T)
    v = zero(Vec{dim, T})
    f = zero(Vec{dim, T})
    return Node(N, m, v, f)
end

@generated function Base.getproperty(node::Node{dim, T}, name::Symbol) where {dim, T}
    return quote
        @_inline_meta
        if name == :x
            return Vec{dim, T}(@inbounds @ntuple $dim d -> node.N.shapes[d].x)
        else
            return getfield(node, name)
        end
    end
end

@inline Base.size(::Node{dim}) where {dim} = (dim,)
@inline @propagate_inbounds Base.getindex(node::Node, i::Int) = node.x[i]
@inline getdirichlet(node::Node) = getdirichlet(node.N)

@inline function reset!(node::Node{dim, T}) where {dim, T}
    node.m = zero(T)
    node.v = zero(Vec{dim, T})
    node.f = zero(Vec{dim, T})
    return node
end
