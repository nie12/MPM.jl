module MPM

using Base: @propagate_inbounds, @_inline_meta
using Base.Cartesian: @ntuple
using LinearAlgebra
using TensorArrays
using RecipesBase
using ProgressMeter: @showprogress

export Node, Grid, Particle
export Problem, BoundaryVelocity, BoundaryForce
export UpdateStressFirst
export generategrid, generateparticles, solve

include("utils.jl")
include("particle.jl")
include("shapefunction.jl")
include("node.jl")
include("grid.jl")
include("nodalvalues.jl")
include("boundary.jl")
include("problem.jl")
include("algorithms.jl")
include("solution.jl")
include("output.jl")

end # module
