struct BoundaryVelocity
    v::Function
    nodeinds::Vector{Int}
end

struct BoundaryForce
    f::Function
    nodeinds::Vector{Int}
end

for BoundaryType in (:BoundaryVelocity, :BoundaryForce)
    @eval begin
        @inline function $BoundaryType(f::Function, node::Node)
            $BoundaryType(f, Int[node.id])
        end
        @inline function $BoundaryType(f::Function, nodes::AbstractArray{<: Node})
            $BoundaryType(f, vec(Int[node.id for node in nodes]))
        end
        @inline Base.eachindex(b::$BoundaryType) = b.nodeinds
    end
end
