@testset "Interpolation" begin
    for interp in (Tent,)
        for T in (Float64, Float32), _ in 1:100
            ax = LinRange{T}(0, 1, 5)
            L = step(ax)
            xp = rand(T)
            lp = rand(T)
            shapes = [MPM.LineShape{interp, T}(ax[1], L, MPM.FIXED),
                      MPM.LineShape{interp, T}(ax[2], L, MPM.LFIXED),
                      MPM.LineShape{interp, T}(ax[3], L, MPM.FREE),
                      MPM.LineShape{interp, T}(ax[4], L, MPM.RFIXED),
                      MPM.LineShape{interp, T}(ax[5], L, MPM.FIXED)]
            @test sum(MPM.value(shapes[i], xp, lp) for i in MPM.neighbor_range(interp, ax, xp)) ≈ 1 atol=1e-5
            @test sum(MPM.derivative(shapes[i], xp, lp) for i in MPM.neighbor_range(interp, ax, xp)) ≈ 0 atol=1e-5
        end
    end
    for interp in (Tent, uGIMP,)
        for T in (Float64, Float32), _ in 1:100
            xv = rand(T)
            L = one(T)
            xp = rand(T)
            lp = rand(T)
            shape = @inferred MPM.LineShape{interp}(xv, L)
            @test MPM.derivative(shape, xp, lp) ≈ TensorArrays.gradient(xp -> MPM.value(shape, xp, lp), xp)
            @test MPM.derivative(shape,  L + xp, lp) ≈ TensorArrays.gradient(xp -> MPM.value(shape,  L + xp, lp),  xp)
            @test MPM.derivative(shape, -L - xp, lp) ≈ TensorArrays.gradient(xp -> MPM.value(shape, -L + xp, lp), -xp)
        end
    end
end

@testset "ShapeFunction" begin
    @testset "LinearInterpolation" begin
        for T in (Float64, Float32)
            xi = zero(T)
            L = 2 * one(T)
            x = rand(T)
            shape = @inferred MPM.LineShape{Tent}(xi, L)
            N = ShapeFunction((shape,))
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
