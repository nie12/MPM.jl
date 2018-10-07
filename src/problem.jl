mutable struct Problem{dim, T, interp}
    update_stress!::Function
    grid::Grid{dim, T, interp}
    tspan::Tuple{T, T}
    gravity::Vec{dim, T}
    bvels::Vector{BoundaryVelocity{dim}}
    bforces::Vector{BoundaryForce{dim}}
end

function Problem(update_stress!, grid::Grid{dim, T}, tspan::Tuple{Real,Real}; gravity = false) where {dim, T}
    Problem(update_stress!,
            grid,
            map(T, tspan),
            gravity == false ? zero(Vec{dim, T}) : convert(Vec{dim, T}, gravity),
            BoundaryVelocity{dim}[],
            BoundaryForce{dim}[])
end

Base.push!(prob::Problem, bvel::BoundaryVelocity) = push!(prob.bvels, bvel)
Base.push!(prob::Problem, bforce::BoundaryForce) = push!(prob.bforces, bforce)
