abstract type AbstractBoundary end

struct FixedBoundary <: AbstractBoundary
    f::Function
end

struct NodalForceBoundary <: AbstractBoundary
    f::Function
end

struct BoundaryCondition{B <: AbstractBoundary, dim}
    bound::B
    nodeinds::Vector{CartesianIndex{dim}}
end

@inline nodeindices(bc::BoundaryCondition) = bc.nodeinds

@inline function (bc::BoundaryCondition{FixedBoundary, dim})(node::Node{dim}, t::Real) where {dim}
    bc.bound.f(node, t)::NTuple{dim, Bool}
end

@inline function (bc::BoundaryCondition{NodalForceBoundary, dim})(node::Node{dim}, t::Real) where {dim}
    bc.bound.f(node, t)::NTuple{dim, Real}
end
