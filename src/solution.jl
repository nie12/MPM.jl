struct Solution{dim, T, Ps <: AbstractArray{<: Particle{dim, T}}, VT <: AbstractVector{T}} <: AbstractVector{NamedTuple{(:t, :particles), Tuple{T, Ps}}}
    grid::Grid{dim, T}
    tᵢ::VT
    particlesᵢ::Vector{Ps}
end

@inline Base.size(sol::Solution) = size(sol.tᵢ)
@inline Base.getindex(sol::Solution, i::Int) = (t=sol.tᵢ[i], particles=sol.particlesᵢ[i])

function Solution(prob::Problem, particles_0::AbstractArray{P}; dt = missing, length = missing) where {dim, T, P <: Particle{dim, T}}
    tᵢ = TimeSpan(prob, dt = dt, length = length)
    particlesᵢ = Array{typeof(particles_0)}(undef, size(tᵢ))
    @inbounds for i in eachindex(particlesᵢ)
        particlesᵢ[i] = copy.(particles_0)
    end
    return Solution(prob.grid, tᵢ, particlesᵢ)
end

function solve(prob::Problem, particles::AbstractArray{P}, alg::AbstractAlgorithm; dt::Real, length = missing) where {dim, T, P <: Particle{dim, T}}
    if ismissing(length)
        solve!(Solution(prob, particles, dt = dt), prob, particles, alg; dt = dt)
    else
        solve!(Solution(prob, particles, length = length), prob, particles, alg; dt = dt)
    end
end

function solve!(sol::Solution{dim, T}, prob::Problem{dim, T}, particles::AbstractArray{<: Particle{dim, T}}, alg::AbstractAlgorithm; dt::Real) where {dim, T}
    tspan = TimeSpan(prob, dt = dt)
    particlesₙ = copy.(particles)
    particles = copy.(particles)
    count = 1
    @inbounds @showprogress 0.1 "Computing..." 0 for i in 2:length(tspan)
        tₙ = tspan[i-1]
        t = tspan[i]
        update!(prob, particles, alg, (tₙ, t))
        while checkbounds(Bool, sol.tᵢ, count) && tₙ ≤ sol.tᵢ[count] ≤ t
            interpolate!(sol[count], (tₙ, particlesₙ), (t, particles))
            count += 1
        end
        copy!.(particlesₙ, particles)
    end
    return sol
end

function (sol::Solution{dim, T})(t_::Real) where {dim, T}
    t = T(t_)
    t ≤ sol.tᵢ[1]   && return sol[1]
    t ≥ sol.tᵢ[end] && return sol[end]
    i = findfirst(tᵢ -> t ≤ tᵢ, sol.tᵢ)
    sol.tᵢ[i] == t && return sol[i]
    ξ = (t - sol.tᵢ[i-1]) / (sol.tᵢ[i] - sol.tᵢ[i-1])
    return map((x,y) -> (1-ξ)*x + ξ*y, sol[i-1], sol[i])
end

function interpolate!(sol::NamedTuple{(:t,:particles)}, (tₙ,xₙ), (t,x))
    if sol.t ≤ tₙ
        copy!.(sol.particles, xₙ)
    elseif sol.t ≥ t
        copy!.(sol.particles, x)
    else
        ξ = (sol.t - tₙ) / (t - tₙ)
        @. sol.particles = (1-ξ)*xₙ + ξ*x
    end
    return sol
end


struct TimeSpan{T} <: AbstractVector{T}
    span::LinRange{T}
end

@inline Base.size(tspan::TimeSpan) = size(tspan.span)
@inline Base.getindex(tspan::TimeSpan, i::Int) = tspan.span[i]

@inline function TimeSpan{T}(start::Real, stop::Real; dt = missing, length = missing) where {T}
    @assert !ismissing(dt) ⊻ !ismissing(length)
    if !ismissing(dt)
        span = start:dt:stop
        return span[end] == stop ? TimeSpan{T}(convert(LinRange{T}, span)) :
                                   TimeSpan{T}(LinRange(start, stop, Base.length(span)+1))
    elseif !ismissing(length)
        return TimeSpan{T}(LinRange(start, stop, length))
    end
end
@inline TimeSpan(prob::Problem{dim, T}; dt = missing, length = missing) where {dim, T} = @inbounds TimeSpan{T}(prob.tspan[1], prob.tspan[2], dt = dt, length = length)
