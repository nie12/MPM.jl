macro inboundsret(ex)
    return quote
        @inbounds v = $(esc(ex))
        v
    end
end

function generateaxs(domain::AbstractMatrix{<: Real}, axsize::NTuple{dim, Int}) where {dim}
    promote_shape(size(domain), (dim, 2)) # check size
    ntuple(Val(dim)) do i
        @inbounds begin
            start = domain[i,1]
            stop = domain[i,2]
            @assert start < stop
            return LinRange(start, stop, axsize[i])
        end
    end
end

function _gcd(a::T, b::T) where {T <: Real}
    if a < b
        return gcd_float(b, a)
    end
    if abs(b) < √eps(T)
        return a
    else
        return _gcd(b, a - floor(a / b) * b)
    end
end
_gcd(a::Int, b::Int) = gcd(a, b)
_gcd(x::Real) = x
_gcd(xs::T...) where {T <: Real} = _gcd(_gcd(xs[1], xs[2]), xs[3:end]...)
gcd_float(xs::Real...) = _gcd(promote(xs...)...)
