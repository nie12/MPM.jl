struct Problem{dim, T, interp, F}
    update_stress!::F
    grid::Grid{interp, dim, T}
    tspan::Tuple{T, T}
    gravity::Vec{dim, T}
end

function Problem(update_stress!, grid::Grid{interp, dim, T}, tspan::Tuple{Real,Real}; gravity::Union{Missing, Vec{dim}} = missing) where {interp, dim, T}
    Problem(update_stress!,
            grid,
            map(T, tspan),
            gravity === missing ? zero(Vec{dim, T}) : convert(Vec{dim, T}, gravity))
end

@inline function apply(update_stress!, iₚ, pt::MaterialPoint, dt::Real)
    update_stress!(pt, dt)
end
@inline @propagate_inbounds function apply(update_stress!::Array, iₚ, pt::MaterialPoint, dt::Real)
    update_stress![iₚ](pt, dt)
end
