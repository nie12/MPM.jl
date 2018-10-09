struct FixedBoundary{dim}
    condition::Function
    nodeinds::Vector{CartesianIndex{dim}}
end

struct NodalForceBoundary{dim}
    f::Function
    nodeinds::Vector{CartesianIndex{dim}}
end

for BoundaryType in (:FixedBoundary, :NodalForceBoundary)
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
