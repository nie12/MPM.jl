macro inboundsret(ex)
    return quote
        @inbounds v = $(esc(ex))
        v
    end
end

function generateaxs(domain::AbstractMatrix{T}, axsize::NTuple{dim, Int}) where {T <: Real, dim}
    promote_shape(size(domain), (dim, 2)) # check size
    ntuple(Val(dim)) do i
        @inbounds begin
            start = domain[i,1]
            stop = domain[i,2]
            @assert start < stop
            return LinRange{T}(start, stop, axsize[i])
        end
    end
end

@generated function connectivity(eltindex::NTuple{dim, Int}) where {dim}
    return quote
        @_inline_meta
        @inbounds return CartesianIndices(@ntuple $dim i -> eltindex[i]:eltindex[i]+1)
    end
end
@inline connectivity(eltindex::Vararg{Int}) = connectivity(eltindex)
@inline connectivity(eltindex::CartesianIndex) = connectivity(Tuple(eltindex))

@inline /₀(x, y::Real) = y ≈ zero(y) ? zero(x) : x / y
