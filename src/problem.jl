mutable struct Problem{dim, T, interp}
    update_stress!::Function
    grid::Grid{interp, dim, T}
    tspan::Tuple{T, T}
    gravity::Vec{dim, T}
end

function Problem(update_stress!, grid::Grid{interp, dim, T}, tspan::Tuple{Real,Real}; gravity = false) where {interp, dim, T}
    Problem(update_stress!,
            grid,
            map(T, tspan),
            gravity == false ? zero(Vec{dim, T}) : convert(Vec{dim, T}, gravity))
end
