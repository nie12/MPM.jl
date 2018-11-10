abstract type Algorithm end

struct USF  <: Algorithm end
struct USL  <: Algorithm end
struct MUSL <: Algorithm end

function update!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, ::USF, tspan::Tuple{T, T}) where {dim, T}
    update_dirichlet!(prob.grid, tspan)
    reset!(prob.grid)
    compute_nodal_mass_and_momentum!(prob, pts, tspan)
    update_particle_stress!(prob, pts, tspan)
    compute_nodal_force!(prob, pts, tspan)
    update_particle_position_and_velocity!(prob, pts, tspan)
end
function update!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, ::USL, tspan::Tuple{T, T}) where {dim, T}
    update_dirichlet!(prob.grid, tspan)
    reset!(prob.grid)
    compute_nodal_mass_and_momentum!(prob, pts, tspan)
    compute_nodal_force!(prob, pts, tspan)
    update_particle_position_and_velocity!(prob, pts, tspan)
    update_particle_stress!(prob, pts, tspan)
end
function update!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, ::MUSL, tspan::Tuple{T, T}) where {dim, T}
    update_dirichlet!(prob.grid, tspan)
    @inbounds if prob.tspan[1] == tspan[1]
        reset!(prob.grid)
        compute_nodal_mass_and_momentum!(prob, pts, tspan)
    end
    compute_nodal_force!(prob, pts, tspan)
    update_particle_position_and_velocity!(prob, pts, tspan)
    reset!(prob.grid)
    compute_nodal_mass_and_momentum!(prob, pts, tspan)
    update_particle_stress!(prob, pts, tspan)
end

function compute_nodal_mass_and_momentum!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    @inbounds t = tspan[1]
    grid = prob.grid

    #=
    Compute nodal mass and nodal momentum
    =#
    for pt in pts
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            N = node.N(pt)
            node.m += N * pt.m
            node.mv += N * pt.m * pt.v
        end
    end

    #=
    Modify nodal momentum for fixed boundary condition
    =#
    for bc in grid.dirichlets
        for i in nodeindices(bc)
            @inbounds node = grid[i]
            cond = getdirichlet(node)
            node.mv = Vec{dim, T}(i -> cond[i] == FIXED ? zero(T) : node.mv[i])
        end
    end
end

function update_particle_stress!(prob::Problem{dim, T, interp}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {interp, dim, T}
    @inbounds dt = tspan[2] - tspan[1]
    grid = prob.grid

    for (iₚ, pt) in enumerate(pts)
        L = zero(pt.L)
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            if node.m > √eps(T)
                N′ = node.N'(pt)
                v = node.mv / node.m
                L += v ⊗ N′
            end
        end
        pt.L = L
        pt.F += dt*L ⋅ pt.F
        apply(prob.update_stress!, iₚ, pt, dt)
        update_particle_domain!(pt, interp)
    end
end

function compute_nodal_force!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    @inbounds t = tspan[1]
    grid = prob.grid

    #=
    Compute internal force and gravity force
    =#
    for pt in pts
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            N′, N = node.N'(pt, :all)
            fint = (-det(pt.F) * pt.m / pt.ρ₀) * (pt.σ ⋅ N′)
            fext = (N * pt.m) * prob.gravity
            node.f += fint + fext
        end
    end

    #=
    Compute surface force by Neumann boundary condition
    =#
    for bc in grid.neumanns
        for i in nodeindices(bc)
            @inbounds node = grid[i]
            f = bc(node, t)
            node.f += Vec{dim, T}(i -> f[i] === missing ? node.f[i] : f[i])
        end
    end

    #=
    Modify nodal force for fixed boundary condition
    (assume that acceleration is zero)
    =#
    for bc in grid.dirichlets
        for i in nodeindices(bc)
            @inbounds node = grid[i]
            cond = getdirichlet(node)
            node.f = Vec{dim, T}(i -> cond[i] == FIXED ? zero(T) : node.f[i])
        end
    end
end

function update_particle_position_and_velocity!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    @inbounds dt = tspan[2] - tspan[1]
    grid = prob.grid

    #=
    Update nodal momentum
    =#
    for node in grid
        node.mv += dt * node.f
    end

    #=
    Update velocity and position for material points
    =#
    for pt in pts
        a = zero(pt.v)
        v = zero(pt.x)
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            if node.m > √eps(T)
                N = node.N(pt)
                a += (N / node.m) * node.f
                v += (N / node.m) * node.mv
            end
        end
        pt.v += dt * a
        pt.x += dt * v
    end
end
