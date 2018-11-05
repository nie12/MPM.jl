module MPM

using Base: @propagate_inbounds, @_inline_meta
using Base.Cartesian: @ntuple, @nexprs
using LinearAlgebra
using TensorArrays
using RecipesBase
using ProgressMeter: @showprogress

export ShapeFunction, Tent, uGIMP, cpGIMP
export Node, Grid, MaterialPoint
export Problem, DirichletBoundary, NeumannBoundary
export USF, USL, MUSL
export generategrid, generatepoints, solve, add!
export FREE, FIXED

include("utils.jl")
include("materialpoint.jl")
include("shapefunction.jl")
include("node.jl")
include("boundary.jl")
include("grid.jl")
include("problem.jl")
include("algorithms.jl")
include("snapshot.jl")
include("solution.jl")
include("output.jl")

end # module
