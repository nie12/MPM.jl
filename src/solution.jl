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
    axs::NTuple{dim, LinRange{T}}
    t_::TimeSpan{T}
    pts_::Vector{Array{MaterialPoint{dim, T, M, Ms}, N}}
end

@inline Base.size(sol::Solution) = size(sol.t_)
@inline function Base.getindex(sol::Solution, i::Int)
    steps = step.(sol.axs)
    limits = extrema.(sol.axs)
    SnapShot(steps, limits, sol.t_[i], sol.pts_[i])
end

function Solution(prob::Problem, pts::AbstractArray{MP}; dt = missing, length = missing) where {dim, T, MP <: MaterialPoint{dim, T}}
    t_ = TimeSpan(prob, dt = dt, length = length)
    pts_ = Array{typeof(pts)}(undef, size(t_))
    @inbounds for i in eachindex(pts_)
        pts_[i] = copy.(pts)
    end
    return Solution(prob.grid.axs, t_, pts_)
end

function solve(prob::Problem, pts::Array{<: MaterialPoint}, alg::Algorithm; dt::Real, length = missing)
    if ismissing(length)
        solve!(Solution(prob, pts, dt = dt), prob, pts, alg; dt = dt)
    else
        solve!(Solution(prob, pts, length = length), prob, pts, alg; dt = dt)
    end
end

function solve!(sol::Solution{dim, T}, prob::Problem{dim, T}, points::Array{<: MaterialPoint{dim, T}}, alg::Algorithm; dt::Real) where {dim, T}
    tspan = TimeSpan(prob, dt = dt)
    ptsₙ = copy.(points)
    pts = copy.(points)
    count = 1
    @inbounds @showprogress 0.1 "Computing..." 0 for i in 2:length(tspan)
        tₙ = tspan[i-1]
        t = tspan[i]
        update!(prob, pts, alg, (tₙ, t))
        while checkbounds(Bool, sol.t_, count) && tₙ ≤ sol.t_[count] ≤ t
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
    i = findfirst(t_ -> t < t_, sol.t_)
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
