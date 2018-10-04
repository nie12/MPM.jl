abstract type AbstractAlgorithm end

struct UpdateStressFirst <: AbstractAlgorithm end

function update!(prob::Problem{dim, T}, pts::AbstractArray{<: MaterialPoint{dim, T}}, ::UpdateStressFirst, tspan::Tuple{Real,Real}) where {dim, T}
    dt = tspan[2] - tspan[1]
    t = tspan[1]
    grid = reset!(prob.grid)

    #=
    Compute nodal mass and nodal momentum
    =#
    for pt in pts
        mᵢ = NodalValues((node,pt) -> node.N(pt)*pt.m, grid, pt)
        mvᵢ = NodalValues((node,pt) -> node.N(pt)*(pt.m*pt.v), grid, pt)
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
    for pt in pts
        pt.L = sum(NodalValues((node,pt) -> node.v ⊗ node.N'(pt), grid, pt))
        pt.F = (I + dt*pt.L) ⋅ pt.F
        prob.update_stress!(pt, dt) # TODO: consider better way to avoid type instability
        fintᵢ = NodalValues((node,pt) -> (-det(pt.F)*pt.m/pt.ρ₀) * pt.σ ⋅ node.N'(pt), grid, pt)
        @inbounds add!(grid.f, fintᵢ)
    end

    #=
    Add external nodal force for gravity and Neumann boundary condition
    =#
    for pt in pts
        fextᵢ = NodalValues((node,pt) -> node.N(pt)*pt.m*prob.gravity, grid, pt)
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
    Update velocity and position for material points
    =#
    for pt in pts
        pt.v = pt.v + dt * sum(NodalValues((node,pt) -> node.N(pt) * node.f /₀ node.m, grid, pt))
        pt.x = pt.x + dt * sum(NodalValues((node,pt) -> node.N(pt) * node.mv /₀ node.m, grid, pt))
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
