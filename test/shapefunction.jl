@testset "ShapeFunction" begin
    for T in (Float64, Float32)
        xi = zero(T)
        l = 2 * one(T)
        x = rand(T)
        shape = ShapeFunction(xi, l)
        @test (@inferred value(shape,  x))::T ≈ (1 - x / l)
        @test (@inferred value(shape, -x))::T ≈ (1 - x / l)
        @test (@inferred gradient(shape,  x))::T ≈ -1 / l
        @test (@inferred gradient(shape, -x))::T ≈  1 / l
        @test (@inferred value(shape,  x + l))::T ≈ zero(T)
        @test (@inferred value(shape, -x - l))::T ≈ zero(T)
        @test (@inferred gradient(shape,  x + l))::T ≈ zero(T)
        @test (@inferred gradient(shape, -x - l))::T ≈ zero(T)
    end
end
