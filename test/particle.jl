@testset "Particle" begin
    for T in (Float32, Float64)
        p = (@inferred Particle(x = Vec(rand(T)), m = rand(T), V₀ = rand(T)))::Particle{1, T, 1, 1}
        # +p, -p
        for op in (+, -)
            y = (@inferred op(p))::typeof(p)
            for name in fieldnames(Particle)
                @test getproperty(y, name) == op(getproperty(p, name))
            end
        end
        # p + p, p - p
        for op in (+, -)
            y = (@inferred op(p, p))::typeof(p)
            for name in fieldnames(Particle)
                @test getproperty(y, name) == op(getproperty(p, name), getproperty(p, name))
            end
        end
        # 2p
        y = (@inferred 2p)::typeof(p)
        for name in fieldnames(Particle)
            @test getproperty(y, name) == 2getproperty(p, name)
        end
        # p*2, p/2
        for op in (*, /)
            y = (@inferred op(p, 2))::typeof(p)
            for name in fieldnames(Particle)
                @test getproperty(y, name) == op(getproperty(p, name), 2)
            end
        end
    end
end
