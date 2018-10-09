mutable struct Problem{dim, T, interp}
    update_stress!::Function
    grid::Grid{dim, T, interp}
    tspan::Tuple{T, T}
    gravity::Vec{dim, T}
    fixedbc::Vector{FixedBoundary{dim}}
    bforces::Vector{NodalForceBoundary{dim}}
end

function Problem(update_stress!, grid::Grid{dim, T}, tspan::Tuple{Real,Real}; gravity = false) where {dim, T}
    Problem(update_stress!,
            grid,
            map(T, tspan),
            gravity == false ? zero(Vec{dim, T}) : convert(Vec{dim, T}, gravity),
            FixedBoundary{dim}[],
            NodalForceBoundary{dim}[])
end

Base.push!(prob::Problem, bvel::FixedBoundary) = push!(prob.fixedbc, bvel)
Base.push!(prob::Problem, bforce::NodalForceBoundary) = push!(prob.bforces, bforce)
