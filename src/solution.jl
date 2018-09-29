struct Solution{dim, T, P <: AbstractArray{<: Particle{dim, T}}} <: AbstractVector{NamedTuple{(:t, :particles), Tuple{T, P}}}
    grid::Grid{dim, T}
    t::Vector{T}
    particles::Vector{P}
end

@inline Base.size(sol::Solution) = size(sol.t)
@inline Base.getindex(sol::Solution, i::Int) = (t=sol.t[i], particles=sol.particles[i])

function solve(prob::Problem, particles::AbstractArray{<: Particle{dim, T}}, alg::AbstractAlgorithm; dt::Real) where {dim, T}
    ts = prob.tspan[1]:dt:prob.tspan[2]
    nt = length(ts)
    particlesᵢ = Vector{typeof(particles)}(undef, nt)
    tᵢ = Vector{T}(undef, nt)
    particlesᵢ[1] = deepcopy(particles)
    tᵢ[1] = ts[1]
    for i in 2:length(ts)
        particlesᵢ[i] = deepcopy(particlesᵢ[i-1])
        tᵢ[i] = ts[i]
        update!(prob, particlesᵢ[i], alg, (tᵢ[i-1], tᵢ[i]))
    end
    if ts[end] < prob.tspan[2]
        tspan = (ts[end], prob.tspan[2])
        push!(particlesᵢ, deepcopy(particlesᵢ[end]))
        push!(tᵢ, tspan[2])
        update!(prob, particlesᵢ[end], alg, tspan)
    end
    return Solution(prob.grid, tᵢ, particlesᵢ)
end

function (sol::Solution{dim, T})(t_::Real) where {dim, T}
    t = T(t_)
    t ≤ sol.t[1]   && return sol[1]
    t ≥ sol.t[end] && return sol[end]
    i = findfirst(tᵢ -> t < tᵢ, sol.t)
    ξ = (t - sol.t[i-1]) / (sol.t[i] - sol.t[i-1])
    return map((x,y) -> (1-ξ)*x + ξ*y, sol[i-1], sol[i])
end
