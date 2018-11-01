abstract type AbstractBoundary end

struct FixedBoundary <: AbstractBoundary
    f::Function
end
@inline function (b::FixedBoundary)(node::Node{dim}, t::Real) where {dim}
    b.f(node, t)::NTuple{dim, Bool}
end

struct NodalForceBoundary <: AbstractBoundary
    f::Function
end
@inline function (b::NodalForceBoundary)(node::Node{dim}, t::Real) where {dim}
    b.f(node, t)::NTuple{dim, Real}
end

struct BoundaryCondition{BC <: AbstractBoundary, dim} <: AbstractArray{Union{Missing, BC}, dim}
    gridsize::NTuple{dim, Int}
    bcmapping::Dict{Int, BC}
end
@inline function BoundaryCondition{BC}(gridsize::NTuple{dim, Int}) where {BC, dim}
    BoundaryCondition(gridsize, Dict{Int, BC}())
end

@inline Base.size(bc::BoundaryCondition) = bc.gridsize
@inline Base.IndexStyle(::Type{<: BoundaryCondition}) = IndexLinear()

@inline nodeindices(bc::BoundaryCondition) = keys(bc.bcmapping)

@inline function Base.getindex(bc::BoundaryCondition, i::Int)
    get(bc.bcmapping, i, missing)
end
@inline function Base.setindex!(bc::BoundaryCondition, x::AbstractBoundary, i::Int)
    bc.bcmapping[i] = x
end
