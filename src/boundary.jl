abstract type AbstractBoundary end

struct DirichletBoundary <: AbstractBoundary
    f::Function
end

struct NeumannBoundary <: AbstractBoundary
    f::Function
end

struct BoundaryCondition{B <: AbstractBoundary, dim}
    bound::B
    nodeinds::Vector{CartesianIndex{dim}}
end

@inline nodeindices(bc::BoundaryCondition) = bc.nodeinds

@inline function (bc::BoundaryCondition{DirichletBoundary, dim})(node::Node{dim}, t::Real) where {dim}
    bc.bound.f(node, t)::NTuple{dim, Dirichlet}
end

@inline function (bc::BoundaryCondition{NeumannBoundary, dim})(node::Node{dim, T}, t::Real) where {dim, T}
    bc.bound.f(node, t)::Vec{dim, T}
end
