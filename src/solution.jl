struct TimeSpan{T} <: AbstractVector{T}
    span::LinRange{T}
end

@inline Base.size(tspan::TimeSpan) = size(tspan.span)
@inline @propagate_inbounds Base.getindex(tspan::TimeSpan, i::Int) = tspan.span[i]

function TimeSpan{T}(start::Real, stop::Real; dt = missing, length = missing) where {T}
    @assert !ismissing(dt) ⊻ !ismissing(length)
    if !ismissing(dt)
        span = start:dt:stop
        return span[end] == stop ? TimeSpan{T}(convert(LinRange{T}, span)) :
                                   TimeSpan{T}(LinRange(start, stop, Base.length(span)+1))
    elseif !ismissing(length)
        return TimeSpan{T}(LinRange(start, stop, length))
    end
end
function TimeSpan(prob::Problem{dim, T}; dt = missing, length = missing) where {dim, T}
    TimeSpan{T}(prob.tspan[1], prob.tspan[2], dt = dt, length = length)
end


struct Solution{dim, T, N, M, Ms} <: AbstractVector{SnapShot{dim, T, N, M, Ms}}
    grid::Grid{dim, T}
    tᵢ::TimeSpan{T}
    pointsᵢ::Vector{Array{MaterialPoint{dim, T, M, Ms}, N}}
end

@inline Base.size(sol::Solution) = size(sol.tᵢ)
@generated function Base.getindex(sol::Solution{dim}, i::Int) where {dim}
    return quote
        steps = @ntuple $dim i -> stepaxis(sol.grid, i)
        limits = @ntuple $dim i -> (minaxis(sol.grid, i), maxaxis(sol.grid, i))
        SnapShot(steps, limits, sol.tᵢ[i], sol.pointsᵢ[i])
    end
end

function Solution(prob::Problem, pts::AbstractArray{MP}; dt = missing, length = missing) where {dim, T, MP <: MaterialPoint{dim, T}}
    tᵢ = TimeSpan(prob, dt = dt, length = length)
    pointsᵢ = Array{typeof(pts)}(undef, size(tᵢ))
    @inbounds for i in eachindex(pointsᵢ)
        pointsᵢ[i] = copy.(pts)
    end
    return Solution(prob.grid, tᵢ, pointsᵢ)
end

function solve(prob::Problem, pts::Array{<: MaterialPoint}, alg::AbstractAlgorithm; dt::Real, length = missing)
    if ismissing(length)
        solve!(Solution(prob, pts, dt = dt), prob, pts, alg; dt = dt)
    else
        solve!(Solution(prob, pts, length = length), prob, pts, alg; dt = dt)
    end
end

function solve!(sol::Solution{dim, T}, prob::Problem{dim, T}, points::Array{<: MaterialPoint{dim, T}}, alg::AbstractAlgorithm; dt::Real) where {dim, T}
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
    t ≤ sol[1].t   && return sol[1]
    t ≥ sol[end].t && return sol[end]
    i = findfirst(tᵢ -> t < tᵢ, sol.tᵢ)
    out = similar(sol[1])
    out.t = t
    interpolate!(out, (sol[i-1].t, sol[i-1].points), (sol[i].t, sol[i].points))
end

function interpolate!(snap::SnapShot, (tₙ,xₙ), (t,x))
    if snap.t ≤ tₙ
        copy!.(snap.points, xₙ)
    elseif snap.t ≥ t
        copy!.(snap.points, x)
    else
        ξ = (snap.t - tₙ) / (t - tₙ)
        @. snap.points = (1-ξ)*xₙ + ξ*x
    end
    return snap
end
