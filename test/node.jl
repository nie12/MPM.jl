@testset "Node" begin
    for T in (Float64, Float32), dim in 1:3
        coord = fill(-1, Vec{dim, T})
        l = ntuple(i -> 2, Val(dim))
        grid = @inferred generategrid(vcat([[coord[i] coord[i]+l[i]] for i in 1:dim]...), [1 for i in 1:dim]...)
        node = grid[1]
        @test sum(x -> (@inferred node.N(x))::T, gausspoints(T, Val(dim))) ≈ one(T)
        @test sum(x -> (@inferred node.N'(x + coord))::Vec{dim, T}, gausspoints(T, Val(dim))) ≈ zero(Vec{dim, T}) atol=1e-6
    end
end
