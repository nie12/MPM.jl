using LinearAlgebra
using MPM, TensorArrays
using Test

using MPM: value, gradient, ShapeFunction, shape_value, shape_gradient

include("shapefunction.jl")
include("node.jl")
