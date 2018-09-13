@testset "ShapeFunction" begin
    @testset "ShapeFunction1D" begin
        for T in (Float64, Float32)
            xi = zero(T)
            l = 2 * one(T)
            x = rand(T)
            shape = MPM.ShapeFunction1D(xi, l)
            @test (@inferred MPM.value(shape,  x))::T ≈ (1 - x / l)
            @test (@inferred MPM.value(shape, -x))::T ≈ (1 - x / l)
            @test (@inferred MPM.gradient(shape,  x))::T ≈ -1 / l
            @test (@inferred MPM.gradient(shape, -x))::T ≈  1 / l
            @test (@inferred MPM.value(shape,  x + l))::T ≈ zero(T)
            @test (@inferred MPM.value(shape, -x - l))::T ≈ zero(T)
            @test (@inferred MPM.gradient(shape,  x + l))::T ≈ zero(T)
            @test (@inferred MPM.gradient(shape, -x - l))::T ≈ zero(T)
        end
    end
end
