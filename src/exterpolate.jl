struct NodalValues{dim, T, L} <: AbstractArray{T, dim}
    eltindex::CartesianIndex{dim}
    data::NTuple{L, T}
end

@generated function NodalValues(f::Function, grid::Grid{dim}, p::AbstractParticle) where {dim}
    exps = [:(f(grid[conn[$i]], p)) for i in 1:2^dim]
    return quote
        eltindex = whichelement(grid, p)
        conn = connectivity(eltindex)
        return NodalValues(eltindex, tuple($(exps...)))
    end
end

Base.IndexStyle(::Type{<: NodalValues}) = IndexLinear()

@inline Base.size(::NodalValues{dim}) where {dim} = ntuple(_ -> 2, Val(dim))
@inline @propagate_inbounds Base.getindex(nv::NodalValues, i::Int) = nv.data[i]

@inline connectivity(nv::NodalValues) = connectivity(nv.eltindex)

@inline function add!(A::AbstractArray, nv::NodalValues)
    conn = connectivity(nv)
    @boundscheck checkbounds(A, conn)
    @inbounds for i in 1:length(conn)
        A[conn[i]] += nv[i]
    end
end
