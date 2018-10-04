struct Solution{dim, T, MPs <: AbstractArray{<: MaterialPoint{dim, T}}, VT <: AbstractVector{T}} <: AbstractVector{NamedTuple{(:t, :points), Tuple{T, MPs}}}
    grid::Grid{dim, T}
    tᵢ::VT
    ptsᵢ::Vector{MPs}
end

@inline Base.size(sol::Solution) = size(sol.tᵢ)
@inline Base.getindex(sol::Solution, i::Int) = (t=sol.tᵢ[i], points=sol.ptsᵢ[i])

function Solution(prob::Problem, pts::AbstractArray{MP}; dt = missing, length = missing) where {dim, T, MP <: MaterialPoint{dim, T}}
    tᵢ = TimeSpan(prob, dt = dt, length = length)
    ptsᵢ = Array{typeof(pts)}(undef, size(tᵢ))
    @inbounds for i in eachindex(ptsᵢ)
        ptsᵢ[i] = copy.(pts)
    end
    return Solution(prob.grid, tᵢ, ptsᵢ)
end

function solve(prob::Problem, pts::AbstractArray{MP}, alg::AbstractAlgorithm; dt::Real, length = missing) where {dim, T, MP <: MaterialPoint{dim, T}}
    if ismissing(length)
        solve!(Solution(prob, pts, dt = dt), prob, pts, alg; dt = dt)
    else
        solve!(Solution(prob, pts, length = length), prob, pts, alg; dt = dt)
    end
end

function solve!(sol::Solution{dim, T}, prob::Problem{dim, T}, points::AbstractArray{<: MaterialPoint{dim, T}}, alg::AbstractAlgorithm; dt::Real) where {dim, T}
    tspan = TimeSpan(prob, dt = dt)
    ptsₙ = copy.(points)
    pts = copy.(points)
    count = 1
    @inbounds @showprogress 0.1 "Computing..." 0 for i in 2:length(tspan)
        tₙ = tspan[i-1]
        t = tspan[i]
        update!(prob, pts, alg, (tₙ, t))
        while checkbounds(Bool, sol.tᵢ, count) && tₙ ≤ sol.tᵢ[count] ≤ t
            interpolate!(sol[count], (tₙ, ptsₙ), (t, pts))
            count += 1
        end
        copy!.(ptsₙ, pts)
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

function interpolate!(sol::NamedTuple{(:t,:points)}, (tₙ,xₙ), (t,x))
    if sol.t ≤ tₙ
        copy!.(sol.points, xₙ)
    elseif sol.t ≥ t
        copy!.(sol.points, x)
    else
        ξ = (sol.t - tₙ) / (t - tₙ)
        @. sol.points = (1-ξ)*xₙ + ξ*x
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
