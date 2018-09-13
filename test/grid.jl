@testset "Grid" begin
    for T in (Float32, Float64)
        grid = generategrid(T[0 1], 2)
        @test size(grid) == (3,)

        grid = generategrid(T[0 1; 0 2], 2, 4)
        @test size(grid) == (3, 5)
    end
end
