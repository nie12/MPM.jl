struct NodalValues{dim, T, L}
    eltindex::CartesianIndex{dim}
    data::NTuple{L, T}
end

@generated function NodalValues(f::Function, grid::Grid{dim}, pt::Union{Vec{dim}, MaterialPoint{dim}}) where {dim}
    return quote
        eltindex = whichelement(grid, pt)
        conn = connectivity(eltindex)
        @boundscheck checkbounds(grid, conn)
        @inbounds return NodalValues(eltindex, @ntuple $(2^dim) i -> f(grid[conn[i]], pt))
    end
end

@inline connectivity(nv::NodalValues) = connectivity(nv.eltindex)

@inline function add!(A::AbstractArray, nv::NodalValues)
    conn = connectivity(nv)
    @boundscheck checkbounds(A, conn)
    @inbounds for i in 1:length(conn)
        A[conn[i]] += nv.data[i]
    end
end

@generated function Base.sum(nv::NodalValues{dim, T, L}) where {dim, T, L}
    exps = [:(nv.data[$i]) for i in 1:L]
    return quote
        @_inline_meta
        @inbounds return +($(exps...))
    end
end
