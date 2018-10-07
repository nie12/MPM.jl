abstract type AbstractAlgorithm end

struct USF <: AbstractAlgorithm end

function update!(prob::Problem{dim, T}, pts::AbstractArray{<: MaterialPoint{dim, T}}, ::USF, tspan::Tuple{Real,Real}) where {dim, T}
    t = tspan[1]
    dt = tspan[2] - tspan[1]
    reset!(prob.grid)
    compute_nodal_mass_momentum_velocity!(prob, pts, tspan)
    update_stress!(prob, pts, tspan)
    compute_nodal_force!(prob, pts, tspan)
    update_nodal_mass_momentum_velocity!(prob, pts, tspan)
end

function compute_nodal_mass_momentum_velocity!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    t = tspan[1]
    grid = prob.grid

    #=
    Compute nodal mass and nodal momentum
    =#
    for pt in pts
        for i in relnodeindices(grid, pt)
            @inbounds node = grid[i]
            N = node.N(pt)
            node.m += N * pt.m
            node.mv += N * pt.m * pt.v
        end
    end

    #=
    Compute nodal velocity
    =#
    for node in grid
        if node.m > eps(T)
            node.v = node.mv / node.m
        end
    end

    #=
    Modify nodal velocity for Dirichlet boundary condition
    =#
    for bc in prob.bvels
        @inbounds for i in nodeindices(bc)
            node = grid[i]
            v = bc.v(node, t)
            node.v = Vec(fill_missing(Tuple(node.v), v))
            node.mv = node.m * node.v
        end
    end
end

function update_stress!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    dt = tspan[2] - tspan[1]
    grid = prob.grid

    for pt in pts
        L = zero(pt.L)
        for i in relnodeindices(grid, pt)
            @inbounds node = grid[i]
            N′ = node.N'(pt)
            L += node.v ⊗ N′
        end
        pt.L = L
        pt.F += dt*L ⋅ pt.F
        prob.update_stress!(pt, dt) # TODO: consider better way to avoid type instability
    end
end

function compute_nodal_force!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    t = tspan[1]
    grid = prob.grid

    #=
    Compute internal force and gravity force
    =#
    for pt in pts
        for i in relnodeindices(grid, pt)
            @inbounds node = grid[i]
            N = node.N(pt)
            N′ = node.N'(pt)
            fint = (-det(pt.F)*pt.m/pt.ρ₀) * pt.σ ⋅ N′
            fext = (N*pt.m)*prob.gravity
            node.f += fint + fext
        end
    end

    #=
    Compute surface force by Neumann boundary condition
    =#
    for bc in prob.bforces
        @inbounds for i in nodeindices(bc)
            node = grid[i]
            f = bc.f(node, t)
            node.f += Vec(fill_missing(zero(T), f))
        end
    end
end

function update_nodal_mass_momentum_velocity!(prob::Problem{dim, T}, pts::Array{<: MaterialPoint{dim, T}}, tspan::Tuple{T, T}) where {dim, T}
    t = tspan[1]
    dt = tspan[2] - tspan[1]
    grid = prob.grid

    #=
    Modify nodal force for Dirichelt boundary condition
    (assume that acceleration is zero)
    =#
    for bc in prob.bvels
        @inbounds for i in nodeindices(bc)
            node = grid[i]
            v = bc.v(node, t)
            node.f = Vec(fill_missing(Tuple(node.f), apply_nonmissing(zero(T), v)))
        end
    end

    #=
    Update nodal momentum
    =#
    @. grid.mv = grid.mv + dt * grid.f

    #=
    Update velocity and position for material points
    =#
    for pt in pts
        a = zero(pt.v)
        v = zero(pt.x)
        for i in relnodeindices(grid, pt)
            @inbounds node = grid[i]
            N = node.N(pt)
            if node.m > eps(T)
                a += N * node.f / node.m
                v += N * node.mv / node.m
            end
        end
        pt.v = pt.v + dt * a
        pt.x = pt.x + dt * v
    end
end

@generated function fill_missing(v::NTuple{N, Real}, x::NTuple{N, Union{Missing, Real}}) where {N}
    return quote
        @_inline_meta
        @inbounds @ntuple $N i -> x[i] === missing ? v[i] : x[i]
    end
end
@generated function fill_missing(v::Real, x::NTuple{N, Union{Missing, Real}}) where {N}
    return quote
        @_inline_meta
        @inbounds @ntuple $N i -> x[i] === missing ? v : x[i]
    end
end

@generated function apply_nonmissing(v::Real, x::NTuple{N, Union{Missing, Real}}) where {N}
    return quote
        @_inline_meta
        @inbounds @ntuple $N i -> x[i] === missing ? missing : v
    end
end
