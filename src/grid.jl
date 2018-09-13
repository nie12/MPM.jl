"""
`Grid` is a subtype of `AbstractArray` which stores nodes with linear nodal spacing.
See `generategrid` to construct `Grid` type.
"""
struct Grid{dim, T <: Real} <: AbstractArray{Node{dim, T}, dim}
    axs::NTuple{dim, LinRange{T}}
    m::Array{T, dim}
    v::Array{Vec{dim, T}, dim}
    mv::Array{Vec{dim, T}, dim}
    f::Array{Vec{dim, T}, dim}
end

Base.IndexStyle(::Type{<: Grid}) = IndexCartesian()

@inline Base.size(grid::Grid) = map(length, grid.axs)
@inline function Base.getindex(grid::Grid{dim, T}, cartesian::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(grid, cartesian...)
    N = ShapeFunction(Vec(getindex.(grid.axs, cartesian)), step.(grid.axs))
    @inbounds i = LinearIndices(grid)[cartesian...]
    return Node(i, N,
                pointer(grid.m, i),
                pointer(grid.v, i),
                pointer(grid.mv, i),
                pointer(grid.f, i))
end

"""
    generategrid(domain::AbstractMatrix{<: Real}, nelts...)

Construct `Grid` in the given `domain`. `nelts` is number of elements in each direction.
Use `[0 1; 0 2]` for domain ``[0, 1] \\times [0, 2]``.

# Examples
```jldoctest
julia> generategrid([0 1], 2)
3-element Grid{1,Float64}:
 [0.0]
 [0.5]
 [1.0]

julia> generategrid([0 1; 0 2], 2, 4)
3×5 Grid{2,Float64}:
 [0.0, 0.0]  [0.0, 0.5]  [0.0, 1.0]  [0.0, 1.5]  [0.0, 2.0]
 [0.5, 0.0]  [0.5, 0.5]  [0.5, 1.0]  [0.5, 1.5]  [0.5, 2.0]
 [1.0, 0.0]  [1.0, 0.5]  [1.0, 1.0]  [1.0, 1.5]  [1.0, 2.0]
```
"""
function generategrid(domain::AbstractMatrix{T}, nelts::Vararg{Int, dim}) where {T, dim}
    nnodes = map(i -> i + 1, nelts)
    Grid(generateaxs(domain, nnodes),
         fill(zero(T), nnodes),
         fill(zero(Vec{dim, T}), nnodes),
         fill(zero(Vec{dim, T}), nnodes),
         fill(zero(Vec{dim, T}), nnodes))
end

@inline @propagate_inbounds minaxis(grid::Grid, d::Int) = first(grid.axs[d])
@inline @propagate_inbounds maxaxis(grid::Grid, d::Int) = last(grid.axs[d])
@inline @propagate_inbounds stepaxis(grid::Grid, d::Int) = step(grid.axs[d])

@inline nelements(grid::Grid) = map(i -> i - 1, size(grid))
@inline eachelement(grid::Grid) = CartesianIndices(nelements(grid))

"""
    whichelement(::Grid, p::Vec)
    whichelement(::Grid, p::Particle)

Return the element index in which the particle `p` is located.
"""
@generated function whichelement(grid::Grid{dim}, p::Vec{dim}) where {dim}
    return quote
        @_inline_meta
        @inbounds return CartesianIndex(@ntuple $dim i -> whichelement(minaxis(grid, i), stepaxis(grid, i), p[i]))
    end
end
@inline whichelement(grid::Grid, p::Particle) = whichelement(grid, p.x)
@inline whichelement(x_min::Real, Δx::Real, x::Real) = floor(Int, (x - x_min) / Δx) + 1

function reset!(grid::Grid{dim, T}) where {dim, T}
    fill!(grid.m, zero(T))
    fill!(grid.v, zero(Vec{dim, T}))
    fill!(grid.mv, zero(Vec{dim, T}))
    fill!(grid.f, zero(Vec{dim, T}))
    return grid
end
