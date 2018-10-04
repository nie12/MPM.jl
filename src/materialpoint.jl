mutable struct MaterialPoint{dim, T, M, Ms}
    x  :: Vec{dim, T}                    # coordinate
    ρ₀ :: T                              # initial density
    m  :: T                              # mass
    v  :: Vec{dim, T}                    # velocity
    L  :: Tensor{2, dim, T, M}           # velocity gradient
    F  :: Tensor{2, dim, T, M}           # deformation gradient
    σ  :: SymmetricTensor{2, dim, T, Ms} # stress

    function MaterialPoint{dim, T, M, Ms}() where {dim, T, M, Ms}
        new{dim, T, M, Ms}()
    end
    function MaterialPoint{dim, T, M, Ms}(x, ρ₀, m, v, L, F, σ) where {dim, T, M, Ms}
        new{dim, T, M, Ms}(x, ρ₀, m, v, L, F, σ)
    end
end

@generated function MaterialPoint{dim, T}() where {dim, T}
    M = dim^dim
    Ms = sum(1:dim)
    return quote
        @_inline_meta
        return MaterialPoint{dim, T, $M, $Ms}()
    end
end

@generated function MaterialPoint(x::Vec{dim, <: Real},
                                  ρ₀::Real,
                                  m::Real,
                                  v::Vec{dim, <: Real},
                                  L::Tensor{2, dim, <: Real, M},
                                  F::Tensor{2, dim, <: Real, M},
                                  σ::SymmetricTensor{2, dim, <: Real, Ms}) where {dim, M, Ms}
    T = promote_type(eltype.((x, ρ₀, m, v, L, F, σ))...)
    return quote
        @_inline_meta
        MaterialPoint{dim, $T, M, Ms}(x, ρ₀, m, v, L, F, σ)
    end
end

@inline function MaterialPoint(; x::Vec{dim},
                               ρ₀::Real,
                               m::Real = NaN,
                               v::Vec{dim} = zero(x),
                               L::Tensor{2, dim} = zero(x ⊗ x),
                               F::Tensor{2, dim} = one(L),
                               σ::SymmetricTensor{2, dim} = zero(symmetric(L))) where {dim}
    MaterialPoint(x, ρ₀, m, v, L, F, σ)
end

@generated function Base.convert(::Type{MaterialPoint{dim, T}}, pt::MaterialPoint{dim, U, M, Ms}) where {dim, T, U, M, Ms}
    exps = [:(pt.$name) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        MaterialPoint{dim, T, M, Ms}($(exps...))
    end
end

function generatepoints(f::Function,
                        domain::AbstractMatrix{<: Real},
                        nparts::Vararg{Int, dim};
                        fillbounds::Bool = false) where {dim}
    axs = generateaxs(domain, fillbounds == true ? nparts : map(i->i+2, nparts))
    sz = map(len -> fillbounds == true ? (1:len) : (2:len-1), length.(axs))
    particles = map(CartesianIndices(sz)) do cartesian
        coord = Vec(getindex.(axs, Tuple(cartesian)))
        pt = f(coord)
        T = eltype(eltype(axs))
        return convert(MaterialPoint{dim, T}, pt)
    end
    V = prod(domain[:,2] - domain[:,1])
    np = length(particles)
    Vₚ = V / np
    for pt in particles
        pt.m = pt.ρ₀ * Vₚ
    end
    return particles
end

@inline Base.:+(pt::MaterialPoint) = pt
@generated function Base.:-(pt::MaterialPoint)
    exps = [:(-pt.$name) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        MaterialPoint($(exps...))
    end
end

for op in (:+, :-)
    @eval @generated function Base.$op(x::MaterialPoint, y::MaterialPoint)
        exps = [:($($op)(x.$name, y.$name)) for name in fieldnames(MaterialPoint)]
        return quote
            @_inline_meta
            MaterialPoint($(exps...))
        end
    end
end

for op in (:*, :/)
    @eval @generated function Base.$op(pt::MaterialPoint, x::Real)
        exps = [:($($op)(pt.$name, x)) for name in fieldnames(MaterialPoint)]
        return quote
            @_inline_meta
            MaterialPoint($(exps...))
        end
    end
end
@generated function Base.:*(x::Real, pt::MaterialPoint)
    exps = [:(x * pt.$name) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        MaterialPoint($(exps...))
    end
end

@generated function Base.:(==)(x::MaterialPoint, y::MaterialPoint)
    exps = [:(x.$name == y.$name) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        *($(exps...))
    end
end

@generated function Base.isapprox(x::MaterialPoint, y::MaterialPoint; kwargs...)
    exps = [:(isapprox(x.$name, y.$name; kwargs...)) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        *($(exps...))
    end
end

@generated function Base.copy!(x::MaterialPoint, y::MaterialPoint)
    exps = [:(x.$name = y.$name) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        $(Expr(:block, exps...))
        return x
    end
end
@inline Base.copy(pt::MaterialPoint) = copy!(typeof(pt)(), pt)
