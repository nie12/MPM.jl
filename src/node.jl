struct Node{dim, T, interp} <: AbstractVector{T}
    cartesian::CartesianIndex{dim}
    N::ShapeFunction{interp, dim, T}
    m::Array{T, dim}
    mv::Array{Vec{dim, T}, dim}
    f::Array{Vec{dim, T}, dim}
end

@inline Base.size(::Node{dim}) where {dim} = (dim,)
@inline @propagate_inbounds Base.getindex(node::Node, i::Int) = node.N.x[i]

@inline function Base.getproperty(node::Node, name::Symbol)
    @inbounds begin
        if name == :m
            getfield(node, :m)[getfield(node, :cartesian)]
        elseif name == :mv
            getfield(node, :mv)[getfield(node, :cartesian)]
        elseif name == :f
            getfield(node, :f)[getfield(node, :cartesian)]
        elseif name == :cartesian
            getfield(node, :cartesian)
        elseif name == :N
            getfield(node, :N)
        else
            error("type Node has no field $name")
        end
    end
end

@inline function Base.setproperty!(node::Node, name::Symbol, v)
    @inbounds begin
        if name == :m
            getfield(node, :m)[getfield(node, :cartesian)] = v
        elseif name == :mv
            getfield(node, :mv)[getfield(node, :cartesian)] = v
        elseif name == :f
            getfield(node, :f)[getfield(node, :cartesian)] = v
        else
            error("type Node is immutable")
        end
    end
end
