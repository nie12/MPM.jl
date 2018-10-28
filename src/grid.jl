"""
`Grid` is a subtype of `AbstractArray` which stores nodes with linear nodal spacing.
See `generategrid` to construct `Grid` type.
"""
struct Grid{dim, T, interp} <: AbstractArray{Node{dim, T, interp}, dim}
    axs::NTuple{dim, LinRange{T}}
    m::Array{T, dim}
    mv::Array{Vec{dim, T}, dim}
    f::Array{Vec{dim, T}, dim}
    fixedbounds::Vector{BoundaryCondition{FixedBoundary, dim}}
    forcebounds::Vector{BoundaryCondition{NodalForceBoundary, dim}}
end

Base.IndexStyle(::Type{<: Grid}) = IndexCartesian()

@inline Base.size(grid::Grid) = map(length, grid.axs)
@inline function Base.getindex(grid::Grid{dim, T, interp}, cartesian::Vararg{Int, dim}) where {dim, T, interp}
    @boundscheck checkbounds(grid, cartesian...)
    N = ShapeFunction{interp}(Vec(getindex.(grid.axs, cartesian)), Vec(step.(grid.axs)))
    return Node(CartesianIndex(cartesian), N, grid.m, grid.mv, grid.f)
end

"""
generategrid(domain::AbstractMatrix{<: Real}, nelts...; interpolation::Interpolation)

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
function generategrid(domain::AbstractMatrix{<: Real},
                      nelts::Vararg{Int, dim};
                      interpolation::Interpolation) where {dim}
    nnodes = map(i -> i + 1, nelts)
    axs = generateaxs(domain, nnodes)
    T = eltype(eltype(axs))
    Grid{dim, T, typeof(interpolation)}(axs,
                                        fill(zero(T), nnodes),
                                        fill(zero(Vec{dim, T}), nnodes),
                                        fill(zero(Vec{dim, T}), nnodes),
                                        BoundaryCondition{FixedBoundary, dim}[],
                                        BoundaryCondition{NodalForceBoundary, dim}[])
end

@inline @propagate_inbounds minaxis(grid::Grid, d::Int) = first(grid.axs[d])
@inline @propagate_inbounds maxaxis(grid::Grid, d::Int) = last(grid.axs[d])
@inline @propagate_inbounds stepaxis(grid::Grid, d::Int) = step(grid.axs[d])

@generated function neighbor_nodeindices(grid::Grid{dim}, pt::MaterialPoint{dim}) where {dim}
    return quote
        @_inline_meta
        @inbounds CartesianIndices(@ntuple $dim i -> begin
                                       min = minaxis(grid, i)
                                       step = stepaxis(grid, i)
                                       rng = clamp.(neighbor_range(min, step, pt.x[i], pt.lp[i]), 1, size(grid, i))
                                       rng[1]:rng[2]
                                   end)
    end
end
@inline function neighbor_range(x_min::Real, Δx::Real, x::Real, lp::Real)
    @inbounds begin
        start = surroundings(x_min, Δx, x - (Δx + lp))[2]
        stop  = surroundings(x_min, Δx, x + (Δx + lp))[1]
        return (start, stop)
    end
end
@inline function surroundings(x_min::Real, Δx::Real, x::Real)
    i = floor(Int, (x - x_min) / Δx) + 1
    return (i, i+1)
end

function reset!(grid::Grid{dim, T}) where {dim, T}
    fill!(grid.m,  zero(T))
    fill!(grid.mv, zero(Vec{dim, T}))
    fill!(grid.f,  zero(Vec{dim, T}))
    return grid
end

function generatepoints(f::Function,
                        grid::Grid{dim, T, interp},
                        domain::AbstractMatrix{<: Real},
                        npts::Vararg{Int, dim}) where {dim, T, interp}
    # find indices of `grid.axs` from given domain
    domaininds = Array{Int}(undef, size(domain))
    map!(domaininds, CartesianIndices(domain)) do cartesian
        ax = grid.axs[cartesian[1]]
        if cartesian[2] == 1
            return findfirst(x -> x ≥ domain[cartesian], ax)
        elseif cartesian[2] == 2
            return length(ax) - findfirst(x -> x ≤ domain[cartesian], reverse(ax)) + 1
        end
    end
    axs = ntuple(Val(dim)) do d
        # extract axs where material points should be generated
        ax = grid.axs[d][domaininds[d,1]:domaininds[d,2]]
        out = Vector{eltype(ax)}(undef, (length(ax)-1)*npts[d])
        count = 1
        for i in 1:length(ax)-1
            # divide 1D cell by number of sub-domain
            rng = LinRange(ax[i], ax[i+1], npts[d]+1)
            for j in 1:length(rng)-1
                # material point should be located at the center of sub-domain
                out[count] = (rng[j] + rng[j+1]) / 2
                count += 1
            end
        end
        return out
    end
    # generate material points
    pts = map(CartesianIndices(length.(axs))) do cartesian
        coord = Vec(getindex.(axs, Tuple(cartesian)))
        pt = f(coord)
        return convert(MaterialPoint{dim, T}, pt)
    end
    # initialize material point mass `m` and particle size `lp`
    V = prod(step.(grid.axs))
    Vₚ = V / prod(npts)
    for pt in pts
        pt.m = pt.ρ₀ * Vₚ
        if interp <: GIMP
            pt.lp₀ = Vec(map(/, step.(grid.axs), 2 .* npts))
            pt.lp = pt.lp₀
        end
    end
    return pts
end

@inline function add!(f::Function, grid::Grid, bound::FixedBoundary)
    bc = BoundaryCondition{FixedBoundary}(size(grid))
    @inbounds for i in eachindex(grid)
        if f(grid[i]) == true
            bc[i] = bound
        end
    end
    push!(grid.fixedbounds, bc)
end
@inline function add!(f::Function, grid::Grid, bound::NodalForceBoundary)
    bc = BoundaryCondition{NodalForceBoundary}(size(grid))
    @inbounds for i in eachindex(grid)
        if f(grid[i]) == true
            bc[i] = bound
        end
    end
    push!(grid.forcebounds, bc)
end
