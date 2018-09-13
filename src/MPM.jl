module MPM

using Base: @propagate_inbounds, @_inline_meta
using Base.Cartesian: @ntuple
using LinearAlgebra
using TensorArrays

export Node, Grid, Particle
export Problem, BoundaryVelocity, BoundaryForce
export generategrid, generateparticles

include("utils.jl")
include("particle.jl")
include("shapefunction.jl")
include("node.jl")
include("grid.jl")
include("nodalvalues.jl")
include("boundary.jl")
include("problem.jl")

end # module
