struct BoundaryVelocity{dim}
    v::Function
    nodeinds::Vector{CartesianIndex{dim}}
end

struct BoundaryForce{dim}
    f::Function
    nodeinds::Vector{CartesianIndex{dim}}
end

for BoundaryType in (:BoundaryVelocity, :BoundaryForce)
    @eval begin
        @inline function $BoundaryType(f::Function, node::Node)
            $BoundaryType(f, [node.cartesian])
        end
        @inline function $BoundaryType(f::Function, nodes::AbstractArray{<: Node})
            $BoundaryType(f, vec([node.cartesian for node in nodes]))
        end
        @inline nodeindices(b::$BoundaryType) = b.nodeinds
    end
end
