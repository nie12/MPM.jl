module MPM

using Base: @propagate_inbounds, @_inline_meta
using LinearAlgebra
using TensorArrays
using ForwardDiff

export Node, Grid, AbstractParticle, SimpleParticle
export generategrid, generateparticles

include("utils.jl")
include("particle.jl")
include("shapefunction.jl")
include("node.jl")
include("grid.jl")

end # module
