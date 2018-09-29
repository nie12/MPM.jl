mutable struct Problem{dim, T <: Real}
    update_stress!::Function
    grid::Grid{dim, T}
    tspan::Tuple{T, T}
    gravity::Vec{dim, T}
    bvels::Vector{BoundaryVelocity}
    bforces::Vector{BoundaryForce}
end

function Problem(update_stress!, grid::Grid{dim, T}, tspan::Tuple{Real,Real}; gravity = false) where {dim, T}
    Problem{dim, T}(update_stress!,
                    grid,
                    map(T, tspan),
                    gravity == false ? zero(Vec{dim, T}) : gravity,
                    BoundaryVelocity[],
                    BoundaryForce[])
end

Base.push!(prob::Problem, bvel::BoundaryVelocity) = push!(prob.bvels, bvel)
Base.push!(prob::Problem, bforce::BoundaryForce) = push!(prob.bforces, bforce)
