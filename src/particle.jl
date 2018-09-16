mutable struct Particle{dim, T, M, Ms} <: AbstractVector{T}
    id :: Int
    x  :: Vec{dim, T}                    # coordinate
    m  :: T                              # mass
    V₀ :: T                              # initial volume
    v  :: Vec{dim, T}                    # velocity
    L  :: Tensor{2, dim, T, M}           # velocity gradient
    F  :: Tensor{2, dim, T, M}           # deformation gradient
    σ  :: SymmetricTensor{2, dim, T, Ms} # stress
end

@inline Base.size(p::Particle) = size(p.x)
@inline @propagate_inbounds Base.getindex(p::Particle, i::Int) = p.x[i]

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
        Particle{dim, $T, M, Ms}(-1, x, m, V₀, v, L, F, σ)
    end
end

function generateparticles(f::Function,
                           domain::AbstractMatrix{<: Real},
                           nparts::Vararg{Int, dim};
                           fillbounds::Bool = false) where {dim}
    axs = generateaxs(domain, fillbounds == true ? nparts : map(i->i+2, nparts))
    T = eltype(eltype(axs))
    sz = map(len -> fillbounds == true ? (1:len) : (2:len-1), length.(axs))
    map(CartesianIndices(sz)) do cartesian
        coord = map((ax, i) -> (@inboundsret ax[i]), axs, Tuple(cartesian))
        return f(Vec{dim, T}(coord))::Particle
    end
end
