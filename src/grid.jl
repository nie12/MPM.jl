"""
`Grid` is a subtype of `AbstractArray` which stores nodes with linear nodal spacing.
See `generategrid` to construct `Grid` type.
"""
struct Grid{interp, dim, T} <: AbstractArray{Node{dim, T, interp}, dim}
    axs::NTuple{dim, LinRange{T}}
    nodes::Array{Node{dim, T, interp}, dim}
    dirichlets::Vector{BoundaryCondition{DirichletBoundary, dim}}
    neumanns::Vector{BoundaryCondition{NeumannBoundary, dim}}
end
function Grid{interp}(axs::NTuple{dim, LinRange{T}}) where {interp, dim, T}
    nodes = map(CartesianIndices(length.(axs))) do cartesian
        xs = getindex.(axs, Tuple(cartesian))
        Ls = step.(axs)
        N = ShapeFunction(LineShape{interp}.(xs, Ls))
        return Node(N)
    end
    Grid{interp, dim, T}(axs, nodes,
                         BoundaryCondition{DirichletBoundary, dim}[],
                         BoundaryCondition{NeumannBoundary, dim}[])
end

Base.IndexStyle(::Type{<: Grid}) = IndexCartesian()

@inline Base.size(grid::Grid) = map(length, grid.axs)
@inline function Base.getindex(grid::Grid{interp, dim}, cartesian::Vararg{Int, dim}) where {interp, dim}
    @boundscheck checkbounds(grid, cartesian...)
    @inbounds grid.nodes[cartesian...]
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
function generategrid(::interp, domain::AbstractMatrix{<: Real}, nelts::Vararg{Int, dim}) where {interp, dim}
    nnodes = map(i -> i + 1, nelts)
    axs = generateaxs(domain, nnodes)
    return Grid{interp}(axs)
end

@generated function neighbor_nodeindices(grid::Grid{interp, dim}, pt::MaterialPoint{dim}) where {interp, dim}
    return quote
        @_inline_meta
        CartesianIndices(@inbounds @ntuple $dim d -> begin
                             ax = grid.axs[d]
                             rng = neighbor_element(interp, whichelement(ax, pt.x[d]))::UnitRange{Int}
                             UnitRange(clamp.((minimum(rng), maximum(rng)+1), 1, length(ax))...)
                         end)
    end
end
@inline function whichelement(ax::LinRange{<: Real}, x::Real)
    floor(Int, (x - minimum(ax)) / step(ax)) + 1
end

@inline reset!(grid::Grid) = reset!.(grid)

function generatepoints(f::Function,
                        grid::Grid{interp, dim, T},
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

@inline function add!(f::Function, grid::Grid, bound::DirichletBoundary)
    nodeinds = [cartesian for cartesian in CartesianIndices(grid) if f(grid[cartesian]) == true]
    push!(grid.dirichlets, BoundaryCondition(bound, nodeinds))
end
@inline function add!(f::Function, grid::Grid, bound::NeumannBoundary)
    nodeinds = [cartesian for cartesian in CartesianIndices(grid) if f(grid[cartesian]) == true]
    push!(grid.neumanns, BoundaryCondition(bound, nodeinds))
end

@generated function update_dirichlet!(grid::Grid{interp, dim, T}, tspan::Tuple{T, T}) where {interp, dim, T}
    return quote
        @inbounds t = tspan[1]
        for node in grid
            setdirichlet!(node.N, @ntuple $dim d -> FREE)
        end
        for bc in grid.dirichlets
            for i in nodeindices(bc)
                @inbounds node = grid[i]
                cond = bc(node, t)
                current = getdirichlet(node.N)
                setdirichlet!(node.N, @inbounds @ntuple $dim d -> cond[d] * current[d])
            end
        end
    end
end
