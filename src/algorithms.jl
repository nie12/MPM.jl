abstract type Algorithm end

struct USF  <: Algorithm end
struct USL  <: Algorithm end
struct MUSL <: Algorithm end

function update!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, alg::USF, tspan::Tuple{T, T}) where {dim, T}
    update_dirichlet!(prob, tspan)
    reset_grid!(prob)
    materialpoint_to_grid!(prob, pts, alg, tspan)
    update_materialpoint_stress!(prob, pts, alg, tspan) # update stress first
    calculate_nodal_force!(prob, pts, alg, tspan)       # calculate nodal force based on the updated stress
    update_nodal_momentum!(prob, pts, tspan)
    grid_to_materialpoint!(prob, pts, alg, tspan)
end
function update!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, alg::USL, tspan::Tuple{T, T}) where {dim, T}
    update_dirichlet!(prob, tspan)
    reset_grid!(prob)
    materialpoint_to_grid!(prob, pts, alg, tspan)
    calculate_nodal_force!(prob, pts, alg, tspan)
    update_nodal_momentum!(prob, pts, tspan)
    grid_to_materialpoint!(prob, pts, alg, tspan)
    update_materialpoint_stress!(prob, pts, alg, tspan) # update stress last
end
function update!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, alg::MUSL, tspan::Tuple{T, T}) where {dim, T}
    # do the same process as USL until `grid_to_materialpoint!`
    update_dirichlet!(prob, tspan)
    reset_grid!(prob)
    materialpoint_to_grid!(prob, pts, alg, tspan)
    calculate_nodal_force!(prob, pts, alg, tspan)
    update_nodal_momentum!(prob, pts, tspan)
    grid_to_materialpoint!(prob, pts, alg, tspan)

    # recalculate nodal momentum before `update_materialpoint_stress!`
    grid = prob.grid
    for node in grid
        node.v = zero(Vec{dim, T})
    end
    for pt in pts
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            N = node.N(pt)
            node.v += N * pt.m * pt.v
        end
    end
    for node in grid
        if node.m > √eps(T)
            node.v = node.v / node.m
        else
            node.v = zero(node.v)
        end
    end
    impose_dirichlet_on_nodal_velocity!(prob, pts)

    update_materialpoint_stress!(prob, pts, alg, tspan)
end

function materialpoint_to_grid!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, ::Algorithm, tspan::Tuple{T, T}) where {dim, T}
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
            node.v += N * pt.m * pt.v
        end
    end
    for node in grid
        if node.m > √eps(T)
            node.v = node.v / node.m
        else
            node.v = zero(node.v)
        end
    end

    impose_dirichlet_on_nodal_velocity!(prob, pts)
end

function update_materialpoint_stress!(prob::Problem{dim, T, interp}, pts::Array{<: MaterialPoint{dim, T}}, ::Algorithm, tspan::Tuple{T, T}) where {interp, dim, T}
    @inbounds dt = tspan[2] - tspan[1]
    grid = prob.grid

    for (iₚ, pt) in enumerate(pts)
        L = zero(pt.L)
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            N′ = node.N'(pt)
            L += node.v ⊗ N′
        end
        pt.L = L
        pt.F += dt*L ⋅ pt.F
        apply(prob.update_stress!, iₚ, pt, dt)
        update_particle_domain!(pt, interp)
    end
end

function calculate_nodal_force!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, ::Algorithm, tspan::Tuple{T, T}) where {dim, T}
    grid = prob.grid

    #=
    Compute internal force and gravity force
    =#
    for pt in pts
        for i in neighbor_nodeindices(grid, pt)
            @inbounds node = grid[i]
            N′, N = node.N'(pt, :all)
            fint = -pt.V * (pt.σ ⋅ N′)
            fext = (N * pt.m) * prob.gravity
            node.f += fint + fext
        end
    end

    impose_neumann!(prob, pts, tspan)
    impose_dirichlet_on_nodal_force!(prob, pts)
end

function update_nodal_momentum!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    @inbounds dt = tspan[2] - tspan[1]
    grid = prob.grid
    for node in grid
        if node.m > √eps(T)
            node.v = node.v + dt * (node.f / node.m)
        else
            node.v = zero(node.v)
        end
    end
end

function grid_to_materialpoint!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, ::Algorithm, tspan::Tuple{T, T}) where {dim, T}
    @inbounds dt = tspan[2] - tspan[1]
    grid = prob.grid

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
                v += N * node.v
            end
        end
        pt.v += dt * a
        pt.x += dt * v
    end
end

function impose_dirichlet_on_nodal_velocity!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}) where {dim, T}
    grid = prob.grid
    for bc in grid.dirichlets
        for i in nodeindices(bc)
            @inbounds node = grid[i]
            cond = getdirichlet(node)
            node.v = Vec{dim, T}(i -> cond[i] == FIXED ? zero(T) : node.v[i])
        end
    end
end

function impose_dirichlet_on_nodal_force!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}) where {dim, T}
    grid = prob.grid
    for bc in grid.dirichlets
        for i in nodeindices(bc)
            @inbounds node = grid[i]
            cond = getdirichlet(node)
            node.f = Vec{dim, T}(i -> cond[i] == FIXED ? zero(T) : node.f[i]) # assume that acceleration is zero
        end
    end
end

function impose_neumann!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    @inbounds t = tspan[1]
    grid = prob.grid
    for bc in grid.neumanns
        for i in nodeindices(bc)
            @inbounds node = grid[i]
            node.f = bc(node, t)
        end
    end
end
