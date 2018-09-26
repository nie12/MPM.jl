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

function update_stress_first!(prob::Problem, dt::Real)
    grid = reset!(prob.grid)
    particles = prob.particles

    #=
    Compute nodal mass and nodal momentum
    =#
    for p in particles
        mᵢ = NodalValues((node,p) -> node.N(p)*p.m, grid, p)
        mvᵢ = NodalValues((node,p) -> node.N(p)*(p.m*p.v), grid, p)
        @inbounds add!(grid.m, mᵢ)
        @inbounds add!(grid.mv, mvᵢ)
    end

    #=
    Compute nodal velocity
    =#
    @. grid.v = grid.mv /₀ grid.m

    #=
    Modify nodal velocity for Dirichlet boundary condition
    =#
    for bc in prob.bvels
        @inbounds for i in nodeindices(bc)
            v = bc.v(grid[i])
            grid.v[i] = v
            grid.mv[i] = grid.m[i] * v
        end
    end

    #=
    Update stress and exterpolate nodal force
    =#
    for p in particles
        p.L = sum(NodalValues((node,p) -> node.v ⊗ node.N'(p), grid, p))
        p.F = (I + dt*p.L) ⋅ p.F
        prob.update_stress!(p, dt) # TODO: consider better way to avoid type instability
        fintᵢ = NodalValues((node,p) -> (-det(p.F)*p.V₀) * p.σ ⋅ node.N'(p), grid, p)
        @inbounds add!(grid.f, fintᵢ)
    end

    #=
    Add external nodal force for gravity and Neumann boundary condition
    =#
    for p in particles
        fextᵢ = NodalValues((node,p) -> node.N(p)*p.m*prob.gravity, grid, p)
        @inbounds add!(grid.f, fextᵢ)
    end
    for bc in prob.bforces
        @inbounds for i in nodeindices(bc)
            grid.f[i] += bc.f(grid[i])
        end
    end

    #=
    Modify nodal force for Dirichelt boundary condition
    (assume that acceleration is zero)
    =#
    for bc in prob.bvels
        @inbounds for i in nodeindices(bc)
            grid.f[i] = zero(grid.f[i])
        end
    end

    #=
    Update nodal momentum
    =#
    @. grid.mv = grid.mv + dt * grid.f

    #=
    Update particle velocity and position
    =#
    for p in particles
        p.v = p.v + dt * sum(NodalValues((node,p) -> node.N(p) * node.f /₀ node.m, grid, p))
        p.x = p.x + dt * sum(NodalValues((node,p) -> node.N(p) * node.mv /₀ node.m, grid, p))
    end
end
