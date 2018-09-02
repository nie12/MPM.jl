struct Node{dim, T <: Real} <: AbstractVector{T}
    shapes::NTuple{dim, ShapeFunction{T}}
end

@inline function Node(coord::Vec{dim}, l::NTuple{dim, Real}) where {dim}
    shapes = ntuple(Val(dim)) do i
        @inbounds return ShapeFunction(coord[i], l[i])
    end
    return Node(shapes)
end

@inline Base.size(::Node{dim}) where {dim} = (dim,)
@inline @propagate_inbounds Base.getindex(node::Node, i::Int) = node.shapes[i].xi
@inline steps(node::Node{dim}) where {dim} = ntuple(i -> @inboundsret(step(node.shapes[i])), Val(dim))

@inline function shape_value(node::Node{dim}, x::Vec{dim}) where {dim}
    Ni = ntuple(Val(dim)) do i
        @inbounds begin
            f = node.shapes[i]
            return value(f, x[i])
        end
    end
    return prod(Ni)
end
@inline shape_value(node::Node, p::AbstractParticle) = shape_value(node, p.x)

@inline function shape_gradient(node::Node{dim}, x::Vec{dim}) where {dim}
    gradient(@inline(function(x) shape_value(node, x) end), x)
end
@inline shape_gradient(node::Node, p::AbstractParticle) = shape_gradient(node, p.x)
