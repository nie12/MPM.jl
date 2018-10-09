"""
`Grid` is a subtype of `AbstractArray` which stores nodes with linear nodal spacing.
See `generategrid` to construct `Grid` type.
"""
struct Grid{dim, T, interp} <: AbstractArray{Node{dim, T, interp}, dim}
    axs::NTuple{dim, LinRange{T}}
    m::Array{T, dim}
    mv::Array{Vec{dim, T}, dim}
    f::Array{Vec{dim, T}, dim}
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
                                        fill(zero(Vec{dim, T}), nnodes))
end

@inline @propagate_inbounds minaxis(grid::Grid, d::Int) = first(grid.axs[d])
@inline @propagate_inbounds maxaxis(grid::Grid, d::Int) = last(grid.axs[d])
@inline @propagate_inbounds stepaxis(grid::Grid, d::Int) = step(grid.axs[d])

@inline nelements(grid::Grid) = map(i -> i - 1, size(grid))
@inline eachelement(grid::Grid) = CartesianIndices(nelements(grid))

"""
    whichelement(::Grid, pt::MaterialPoint)

Return the element index in which the material point `pt` is located.
"""
@generated function whichelement(grid::Grid{dim}, pt::MaterialPoint{dim}) where {dim}
    return quote
        @_inline_meta
        @inbounds return CartesianIndex(@ntuple $dim i -> whichelement(minaxis(grid, i), stepaxis(grid, i), pt.x[i]))
    end
end
@inline whichelement(x_min::Real, Δx::Real, x::Real) = floor(Int, (x - x_min) / Δx) + 1

@generated function relnodeindices(grid::Grid{dim, T, Tent}, pt::MaterialPoint{dim}) where {dim, T}
    return quote
        @_inline_meta
        eltindex = whichelement(grid, pt)
        sz = size(grid)
        @inbounds CartesianIndices(@ntuple $dim i -> begin
                                       start = eltindex[i]
                                       stop = eltindex[i] + 1
                                       (1 < start ? start : 1):(sz[i] > stop ? stop : sz[i])
                                   end)
    end
end
@generated function relnodeindices(grid::Grid{dim, T, <: GIMP}, pt::MaterialPoint{dim}) where {dim, T}
    return quote
        @_inline_meta
        eltindex = whichelement(grid, pt)
        sz = size(grid)
        @inbounds CartesianIndices(@ntuple $dim i -> begin
                                       start = eltindex[i] - 1
                                       stop = eltindex[i] + 2
                                       (1 < start ? start : 1):(sz[i] > stop ? stop : sz[i])
                                   end)
    end
end

function reset!(grid::Grid{dim, T}) where {dim, T}
    fill!(grid.m, zero(T))
    fill!(grid.mv, zero(Vec{dim, T}))
    fill!(grid.f, zero(Vec{dim, T}))
    return grid
end

function generatepoints(f::Function,
                        grid::Grid{dim, T},
                        domain::AbstractMatrix{<: Real},
                        npts::Vararg{Int, dim}) where {dim, T}
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
        pt.lp₀ = Vec(map(/, step.(grid.axs), 2 .* npts))
        pt.lp = pt.lp₀
    end
    return pts
end
