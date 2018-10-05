struct Node{dim, T <: Real, Interpolation <: AbstractInterpolation} <: AbstractVector{T}
    id::Int
    N::ShapeFunction{Interpolation, dim, T}
    m::Ptr{T}
    v::Ptr{Vec{dim, T}}
    mv::Ptr{Vec{dim, T}}
    f::Ptr{Vec{dim, T}}
end

@inline Base.size(::Node{dim}) where {dim} = (dim,)
@inline @propagate_inbounds Base.getindex(node::Node, i::Int) = node.N.x[i]

@inline function Base.getproperty(node::Node, name::Symbol)
    return name == :id || name == :N ? getfield(node, name) : unsafe_load(getfield(node, name))
end
