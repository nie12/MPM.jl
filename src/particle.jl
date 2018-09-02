abstract type AbstractParticle{T <: Real} <: AbstractVector{T} end

@inline Base.size(p::AbstractParticle) = size(p.x)
@inline @propagate_inbounds Base.getindex(p::AbstractParticle, i::Int) = p.x[i]

@inline function update!(p::AbstractParticle, ::Val{name}, dx) where {name}
    setfield!(p, name, getfield(p, name) + dx)
end

function generateparticles(f::Function,
                           domain::AbstractMatrix{<: Real},
                           nparts::Vararg{Int, dim};
                           fillbounds::Bool = false) where {dim}
    axs = generateaxs(domain, fillbounds == true ? nparts : map(i->i+2, nparts))
    sz = map(len -> fillbounds == true ? (1:len) : (2:len-1), length.(axs))
    map(CartesianIndices(sz)) do cartesian
        coord = map((ax, i) -> (@inboundsret ax[i]), axs, Tuple(cartesian))
        return f(Vec{dim}(coord))::AbstractParticle
    end
end


mutable struct SimpleParticle{dim, T, L, Ls} <: AbstractParticle{T}
    x  :: Vec{dim, T}                    # coordinate
    m  :: T                              # mass
    V₀ :: T                              # initial volume
    v  :: Vec{dim, T}                    # velocity
    F  :: Tensor{2, dim, T, L}           # deformation gradient
    σ  :: SymmetricTensor{2, dim, T, Ls} # stress
end

@generated function SimpleParticle(; x::Vec{dim, <: Real},
                                     m::Real,
                                     V₀::Real,
                                     v::Vec{dim, <: Real} = zero(x),
                                     F::Tensor{2, dim, <: Real, L} = one(x ⊗ x),
                                     σ::SymmetricTensor{2, dim, <: Real, Ls} = zero(symmetric(F))) where {dim, L, Ls}
    T = promote_type(eltype.((x, m, v, V₀, F, σ))...)
    return quote
        @_inline_meta
        SimpleParticle{dim, $T, L, Ls}(x, m, V₀, v, F, σ)
    end
end
