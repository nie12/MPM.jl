macro inboundsret(ex)
    return quote
        @inbounds v = $(esc(ex))
        v
    end
end

function generateaxes(domain::AbstractMatrix{<: Real}, axsize::NTuple{dim, Int}) where {dim}
    promote_shape(size(domain), (dim, 2)) # check size
    ntuple(Val(dim)) do i
        start = domain[i,1]
        stop = domain[i,2]
        @assert start < stop
        @inboundsret range(start, stop = stop, length = axsize[i])
    end
end
