function quadpoints(::Type{T}, ::Val{1}) where {T <: Real}
    v = T(sqrt(1/3))
    return (Vec{1}((-v,)),
            Vec{1}(( v,)))
end
function quadpoints(::Type{T}, ::Val{2}) where {T <: Real}
    v = T(sqrt(1/3))
    return (Vec{2}((-v, -v)),
            Vec{2}(( v, -v)),
            Vec{2}(( v,  v)),
            Vec{2}((-v,  v)))
end
function quadpoints(::Type{T}, ::Val{3}) where {T <: Real}
    v = T(sqrt(1/3))
    return (Vec{3}((-v, -v, -v)),
            Vec{3}(( v, -v, -v)),
            Vec{3}(( v,  v, -v)),
            Vec{3}((-v,  v, -v)),
            Vec{3}((-v, -v,  v)),
            Vec{3}(( v, -v,  v)),
            Vec{3}(( v,  v,  v)),
            Vec{3}((-v,  v,  v)))
end

@testset "Node" begin
    for T in (Float64, Float32), dim in 1:3
        o = one(T)
        coord = Vec{dim}(ntuple(i -> -o, Val(dim)))
        l = ntuple(i -> 2o, Val(dim))
        node = Node(coord, l)
        @test sum(x -> (@inferred shape_value(node, x))::T, quadpoints(T, Val(dim))) ≈ one(T)
        @test sum(x -> (@inferred shape_gradient(node, x + coord))::Vec{dim, T}, quadpoints(T, Val(dim))) ≈ zero(Vec{dim, T}) atol=1e-6
    end
end
