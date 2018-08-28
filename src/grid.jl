struct Grid{dim, T <: Real} <: AbstractArray{Node{dim, T}, dim}
    nodes::Array{Node{dim, T}, dim}
end

Base.IndexStyle(::Type{<: Grid}) = IndexLinear()

@inline Base.size(grid::Grid) = size(grid.nodes)
@propagate_inbounds @inline Base.getindex(grid::Grid, i::Int) = grid.nodes[i]

function generategrid(domain::AbstractMatrix{<: Real}, nelts::Vararg{Int, dim}) where {dim}
    axes = generateaxes(domain, map(i -> i + 1, nelts))
    return generategrid(axes...)
end

function generategrid(axes::Vararg{StepRangeLen, dim}) where {dim}
    nodes = map(CartesianIndices(length.(axes))) do cartesian
        shapes = ntuple(Val(dim)) do i
            @inbounds begin
                ax = axes[i]
                idx = cartesian[i]
                return ShapeFunction(ax[idx], step(ax))
            end
        end
        return Node(shapes)
    end
    return Grid(nodes)
end
