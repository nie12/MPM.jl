abstract type AbstractAlgorithm end

struct UpdateStressFirst <: AbstractAlgorithm end

function update!(prob::Problem{dim, T}, particles::AbstractArray{<: Particle{dim, T}}, ::UpdateStressFirst, tspan::Tuple{Real,Real}) where {dim, T}
    dt = tspan[2] - tspan[1]
    t = tspan[1]
    grid = reset!(prob.grid)

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
            v = bc.v(grid[i], t)
            grid.v[i] = Vec(fill_missing(Tuple(grid.v[i]), v))
            grid.mv[i] = grid.m[i] * grid.v[i]
        end
    end

    #=
    Update stress and exterpolate nodal force
    =#
    for p in particles
        p.L = sum(NodalValues((node,p) -> node.v ⊗ node.N'(p), grid, p))
        p.F = (I + dt*p.L) ⋅ p.F
        prob.update_stress!(p, dt) # TODO: consider better way to avoid type instability
        fintᵢ = NodalValues((node,p) -> (-det(p.F)*p.m/p.ρ₀) * p.σ ⋅ node.N'(p), grid, p)
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
            f = bc.f(grid[i], t)
            grid.f[i] += Vec(fill_missing(zero(T), f))
        end
    end

    #=
    Modify nodal force for Dirichelt boundary condition
    (assume that acceleration is zero)
    =#
    for bc in prob.bvels
        @inbounds for i in nodeindices(bc)
            v = bc.v(grid[i], t)
            grid.f[i] = Vec(fill_missing(Tuple(grid.f[i]), apply_nonmissing(zero(T), v)))
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
