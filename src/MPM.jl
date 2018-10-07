module MPM

using Base: @propagate_inbounds, @_inline_meta
using Base.Cartesian: @ntuple
using LinearAlgebra
using TensorArrays
using RecipesBase
using ProgressMeter: @showprogress

export ShapeFunction, Tent, uGIMP
export Node, Grid, MaterialPoint
export Problem, BoundaryVelocity, BoundaryForce
export USF
export generategrid, generatepoints, solve

include("utils.jl")
include("materialpoint.jl")
include("shapefunction.jl")
include("node.jl")
include("grid.jl")
include("boundary.jl")
include("problem.jl")
include("algorithms.jl")
include("snapshot.jl")
include("solution.jl")
include("output.jl")

end # module
