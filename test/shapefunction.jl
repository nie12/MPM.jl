@testset "Interpolation" begin
    for interp in (Tent, uGIMP)
        for T in (Float64, Float32), _ in 1:100
            xv = rand(T)
            L = one(T)
            xp = rand(T)
            lp = rand(T)
            @test MPM.derivative(interp, xv, L, xp, lp) ≈ TensorArrays.gradient(xp -> MPM.value(interp, xv, L, xp, lp), xp)
            @test MPM.derivative(interp, xv, L,  L + xp, lp) ≈ TensorArrays.gradient(xp -> MPM.value(interp, xv, L,  L + xp, lp),  xp)
            @test MPM.derivative(interp, xv, L, -L - xp, lp) ≈ TensorArrays.gradient(xp -> MPM.value(interp, xv, L, -L + xp, lp), -xp)
        end
    end
end

@testset "ShapeFunction" begin
    @testset "LinearInterpolation" begin
        for T in (Float64, Float32)
            xi = zero(T)
            L = 2 * one(T)
            x = rand(T)
            N = ShapeFunction{Tent}(Vec(xi), Vec(L))
            @test (@inferred N(MaterialPoint(x =  Vec(x), ρ₀ = 1)))::T ≈ 1 - x / L
            @test (@inferred N(MaterialPoint(x = -Vec(x), ρ₀ = 1)))::T ≈ 1 - x / L
            @test (@inferred N'(MaterialPoint(x =  Vec(x), ρ₀ = 1)))::Vec{1, T} ≈ Vec(-1 / L)
            @test (@inferred N'(MaterialPoint(x = -Vec(x), ρ₀ = 1)))::Vec{1, T} ≈  Vec(1 / L)
            @test (@inferred N(MaterialPoint(x =  Vec(x)+Vec(L), ρ₀ = 1)))::T ≈ zero(T)
            @test (@inferred N(MaterialPoint(x = -Vec(x)-Vec(L), ρ₀ = 1)))::T ≈ zero(T)
            @test (@inferred N'(MaterialPoint(x =  Vec(x)+Vec(L), ρ₀ = 1)))::Vec{1, T} ≈ Vec(zero(T))
            @test (@inferred N'(MaterialPoint(x = -Vec(x)-Vec(L), ρ₀ = 1)))::Vec{1, T} ≈ Vec(zero(T))
        end
    end
end
