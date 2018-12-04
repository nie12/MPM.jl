@testset "Grid" begin
    for T in (Float32, Float64)
        grid = @inferred generategrid(Tent(), T[0, 1], nelements = (2,))
        @test size(grid) == (3,)
        grid = @inferred generategrid(Tent(), T[0, 1], uniform = (2,))
        @test size(grid) == (3,)

        @inferred (grid -> grid[1].N)(grid)
        @inferred (grid -> grid[1].m)(grid)
        @inferred (grid -> grid[1].v)(grid)
        @inferred (grid -> grid[1].f)(grid)
        @test (@inferred (grid -> grid[1].x)(grid))::Vec{1, T} == zero(Vec{1, T})

        grid = @inferred generategrid(Tent(), T[0, 1], T[0, 2], nelements = (2, 4))
        @test size(grid) == (3, 5)
        grid = @inferred generategrid(Tent(), T[0, 1], T[0, 2], uniform = (2, 4))
        @test size(grid) == (3, 5)

        @inferred (grid -> grid[1].N)(grid)
        @inferred (grid -> grid[1].m)(grid)
        @inferred (grid -> grid[1].v)(grid)
        @inferred (grid -> grid[1].f)(grid)
        @test (@inferred (grid -> grid[1].x)(grid))::Vec{2, T} == zero(Vec{2, T})
    end
end
