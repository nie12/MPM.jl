mutable struct Particle{dim, T, M, Ms}
    x  :: Vec{dim, T}                    # coordinate
    ρ₀ :: T                              # initial density
    m  :: T                              # mass
    v  :: Vec{dim, T}                    # velocity
    L  :: Tensor{2, dim, T, M}           # velocity gradient
    F  :: Tensor{2, dim, T, M}           # deformation gradient
    σ  :: SymmetricTensor{2, dim, T, Ms} # stress

    function Particle{dim, T, M, Ms}() where {dim, T, M, Ms}
        new{dim, T, M, Ms}()
    end
    function Particle{dim, T, M, Ms}(x, ρ₀, m, v, L, F, σ) where {dim, T, M, Ms}
        new{dim, T, M, Ms}(x, ρ₀, m, v, L, F, σ)
    end
end
@generated function Particle(x::Vec{dim, <: Real},
                             ρ₀::Real,
                             m::Real,
                             v::Vec{dim, <: Real},
                             L::Tensor{2, dim, <: Real, M},
                             F::Tensor{2, dim, <: Real, M},
                             σ::SymmetricTensor{2, dim, <: Real, Ms}) where {dim, M, Ms}
    T = promote_type(eltype.((x, ρ₀, m, v, L, F, σ))...)
    return quote
        @_inline_meta
        Particle{dim, $T, M, Ms}(x, ρ₀, m, v, L, F, σ)
    end
end

@inline function Particle(; x::Vec{dim},
                            ρ₀::Real,
                            m::Real = NaN,
                            v::Vec{dim} = zero(x),
                            L::Tensor{2, dim} = zero(x ⊗ x),
                            F::Tensor{2, dim} = one(L),
                            σ::SymmetricTensor{2, dim} = zero(symmetric(L))) where {dim}
    Particle(x, ρ₀, m, v, L, F, σ)
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
    particles = map(CartesianIndices(sz)) do cartesian
        coord = Vec(getindex.(axs, Tuple(cartesian)))
        p = f(coord)
        T = eltype(eltype(axs))
        return convert(Particle{dim, T}, p)
    end
    V = prod(domain[:,2] - domain[:,1])
    np = length(particles)
    Vₚ = V / np
    for p in particles
        p.m = p.ρ₀ * Vₚ
    end
    return particles
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

@generated function Base.copy!(x::Particle, y::Particle)
    exps = [:(x.$name = y.$name) for name in fieldnames(Particle)]
    return quote
        @_inline_meta
        $(Expr(:block, exps...))
        return x
    end
end
@inline Base.copy(p::Particle) = copy!(typeof(p)(), p)
