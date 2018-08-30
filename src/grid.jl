# Grid just store the axis informations instead of generating nodes,
# which may be good for huge grid. But not sure that generating node every time
# when `getindex` is used is high cost or not.
struct Grid{dim, T <: Real} <: AbstractArray{Node{dim, T}, dim}
    axs::NTuple{dim, StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}}}
end

Base.IndexStyle(::Type{<: Grid}) = IndexCartesian()

@inline Base.size(grid::Grid) = map(length, grid.axs)
@propagate_inbounds @inline function Base.getindex(grid::Grid{dim}, cartesian::Vararg{Int, dim}) where {dim}
    Node(map((ax, i) -> ShapeFunction(ax[i], step(ax)), grid.axs, cartesian))
end

@inline function generategrid(domain::AbstractMatrix{<: Real}, nelts::Vararg{Int})
    Grid(generateaxs(domain, map(i -> i + 1, nelts)))
end

