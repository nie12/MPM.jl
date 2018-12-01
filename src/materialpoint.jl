mutable struct MaterialPoint{dim, T, M, Ms}
    x   :: Vec{dim, T}                    # coordinate
    ρ₀  :: T                              # initial density
    m   :: T                              # mass
    v   :: Vec{dim, T}                    # velocity
    L   :: Tensor{2, dim, T, M}           # velocity gradient
    F   :: Tensor{2, dim, T, M}           # deformation gradient
    σ   :: SymmetricTensor{2, dim, T, Ms} # stress
    lp  :: Vec{dim, T}
    lp₀ :: Vec{dim, T}

    function MaterialPoint{dim, T, M, Ms}() where {dim, T, M, Ms}
        new{dim, T, M, Ms}()
    end
    function MaterialPoint{dim, T, M, Ms}(x, ρ₀, m, v, L, F, σ, lp, lp₀) where {dim, T, M, Ms}
        new{dim, T, M, Ms}(x, ρ₀, m, v, L, F, σ, lp, lp₀)
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
                                  σ::SymmetricTensor{2, dim, <: Real, Ms},
                                  lp::Vec{dim, <: Real},
                                  lp₀::Vec{dim, <: Real}) where {dim, M, Ms}
    T = promote_type(eltype.((x, ρ₀, m, v, L, F, σ, lp, lp₀))...)
    return quote
        @_inline_meta
        MaterialPoint{dim, $T, M, Ms}(x, ρ₀, m, v, L, F, σ, lp, lp₀)
    end
end

@inline function MaterialPoint(; x::Vec{dim},
                                 ρ₀::Real,
                                 m::Real = 0,
                                 v::Vec{dim} = zero(x),
                                 L::Tensor{2, dim} = zero(x ⊗ x),
                                 F::Tensor{2, dim} = one(L),
                                 σ::SymmetricTensor{2, dim} = zero(symmetric(L)),
                                 lp::Vec{dim} = zero(x)) where {dim}
    MaterialPoint(x, ρ₀, m, v, L, F, σ, lp, lp)
end

@generated function Base.convert(::Type{MaterialPoint{dim, T}}, pt::MaterialPoint{dim, U, M, Ms}) where {dim, T, U, M, Ms}
    exps = [:(pt.$name) for name in fieldnames(MaterialPoint)]
    return quote
        @_inline_meta
        MaterialPoint{dim, T, M, Ms}($(exps...))
    end
end

@inline function Base.getproperty(pt::MaterialPoint, name::Symbol)
    if name == :V
        F = getfield(pt, :F)
        m = getfield(pt, :m)
        ρ₀ = getfield(pt, :ρ₀)
        V₀ = m / ρ₀
        return V₀ * det(F)
    else
        return getfield(pt, name)
    end
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
