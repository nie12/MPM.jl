@testset "ShapeFunction" begin
    @testset "LinearInterpolation" begin
        for T in (Float64, Float32)
            xi = zero(T)
            l = 2 * one(T)
            x = rand(T)
            shape = ShapeFunction(LinearInterpolation(), Vec(xi), Vec(l))
            @test (@inferred MPM.value(shape,  Vec(x)))::T ≈ 1 - x / l
            @test (@inferred MPM.value(shape, -Vec(x)))::T ≈ 1 - x / l
            @test (@inferred MPM.gradient(shape,  Vec(x)))::Vec{1, T} ≈ Vec(-1 / l)
            @test (@inferred MPM.gradient(shape, -Vec(x)))::Vec{1, T} ≈  Vec(1 / l)
            @test (@inferred MPM.value(shape,  Vec(x) + Vec(l)))::T ≈ zero(T)
            @test (@inferred MPM.value(shape, -Vec(x) - Vec(l)))::T ≈ zero(T)
            @test (@inferred MPM.gradient(shape,  Vec(x) + Vec(l)))::Vec{1, T} ≈ Vec(zero(T))
            @test (@inferred MPM.gradient(shape, -Vec(x) - Vec(l)))::Vec{1, T} ≈ Vec(zero(T))
        end
    end
end
