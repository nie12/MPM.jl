# Grid just store the axis informations instead of generating nodes,
# which may be good for huge grid. But not sure that generating node every time
# when `getindex` is used is high cost or not.
struct Grid{dim, T <: Real} <: AbstractArray{Node{dim, T}, dim}
    axs::NTuple{dim, LinRange{T}}
end

Base.IndexStyle(::Type{<: Grid}) = IndexCartesian()

@inline Base.size(grid::Grid) = map(length, grid.axs)
@inline function Base.getindex(grid::Grid{dim}, cartesian::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, cartesian...)
    Node(map((ax, i) -> ShapeFunction(@inboundsret(ax[i]), step(ax)), grid.axs, cartesian))
end

function generategrid(domain::AbstractMatrix{<: Real}, nelts::Vararg{Int})
    Grid(generateaxs(domain, map(i -> i + 1, nelts)))
end

@inline @propagate_inbounds minaxis(grid::Grid, d::Int) = first(grid.axs[d])
@inline @propagate_inbounds maxaxis(grid::Grid, d::Int) = last(grid.axs[d])
@inline @propagate_inbounds stepaxis(grid::Grid, d::Int) = step(grid.axs[d])

@inline nelements(grid::Grid) = map(i -> i - 1, size(grid))
@inline eachelement(grid::Grid) = CartesianIndices(nelements(grid))

function whichelement(grid::Grid{dim}, p::Vec{dim}) where {dim}
    cartesian = ntuple(Val(dim)) do i
        @inbounds begin
            x = p[i]
            x_min = minaxis(grid, i)
            Δx = stepaxis(grid, i)
            return floor(Int, (x - x_min) / Δx) + 1
        end
    end
    return CartesianIndex(cartesian)
end
@inline whichelement(grid::Grid, p::AbstractParticle) = whichelement(grid, p.x)
