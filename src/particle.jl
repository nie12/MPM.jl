mutable struct Particle{dim, T, M, Ms}
    x  :: Vec{dim, T}                    # coordinate
    m  :: T                              # mass
    V₀ :: T                              # initial volume
    v  :: Vec{dim, T}                    # velocity
    L  :: Tensor{2, dim, T, M}           # velocity gradient
    F  :: Tensor{2, dim, T, M}           # deformation gradient
    σ  :: SymmetricTensor{2, dim, T, Ms} # stress
end

@generated function Particle(; x::Vec{dim, <: Real},
                               m::Real,
                               V₀::Real,
                               v::Vec{dim, <: Real} = zero(x),
                               L::Tensor{2, dim, <: Real, M} = zero(x ⊗ x),
                               F::Tensor{2, dim, <: Real, M} = one(L),
                               σ::SymmetricTensor{2, dim, <: Real, Ms} = zero(symmetric(L))) where {dim, M, Ms}
    T = promote_type(eltype.((x, m, v, V₀, L, F, σ))...)
    return quote
        @_inline_meta
        Particle{dim, $T, M, Ms}(x, m, V₀, v, L, F, σ)
    end
end

@generated function Base.convert(::Type{Particle{dim, T}}, p::Particle{dim, U, M, Ms}) where {dim, T, U, M, Ms}
    exps = [:(p.$name) for name in fieldnames(Particle)]
    return quote
        @_inline_meta
        Particle{dim, T, M, Ms}($(exps...))
    end
end

function generateparticles(f::Function,
                           domain::AbstractMatrix{<: Real},
                           nparts::Vararg{Int, dim};
                           fillbounds::Bool = false) where {dim}
    axs = generateaxs(domain, fillbounds == true ? nparts : map(i->i+2, nparts))
    sz = map(len -> fillbounds == true ? (1:len) : (2:len-1), length.(axs))
    map(CartesianIndices(sz)) do cartesian
        coord = Vec(getindex.(axs, Tuple(cartesian)))
        p = f(coord)
        T = eltype(eltype(axs))
        return convert(Particle{dim, T}, p)
    end
end

@inline Base.:+(p::Particle) = p
@generated function Base.:-(p::Particle)
    exps = [:(-p.$name) for name in fieldnames(Particle)]
    return quote
        @_inline_meta
        Particle($(exps...))
    end
end

for op in (:+, :-)
    @eval @generated function Base.$op(x::Particle, y::Particle)
        exps = [:($($op)(x.$name, y.$name)) for name in fieldnames(Particle)]
        return quote
            @_inline_meta
            Particle($(exps...))
        end
    end
end

for op in (:*, :/)
    @eval @generated function Base.$op(p::Particle, x::Real)
        exps = [:($($op)(p.$name, x)) for name in fieldnames(Particle)]
        return quote
            @_inline_meta
            Particle($(exps...))
        end
    end
end
@generated function Base.:*(x::Real, p::Particle)
    exps = [:(x * p.$name) for name in fieldnames(Particle)]
    return quote
        @_inline_meta
        Particle($(exps...))
    end
end

@generated function Base.:(==)(x::Particle, y::Particle)
    exps = [:(x.$name == y.$name) for name in fieldnames(Particle)]
    return quote
        @_inline_meta
        *($(exps...))
    end
end

@generated function Base.isapprox(x::Particle, y::Particle; kwargs...)
    exps = [:(isapprox(x.$name, y.$name; kwargs...)) for name in fieldnames(Particle)]
    return quote
        @_inline_meta
        *($(exps...))
    end
end
