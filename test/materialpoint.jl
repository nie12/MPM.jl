@testset "MaterialPoint" begin
    for T in (Float32, Float64)
        pt = (@inferred MaterialPoint(x = Vec(rand(T)), m = rand(T), ρ₀ = rand(T)))::MaterialPoint{1, T, 1, 1}
        # +pt, -pt
        for op in (+, -)
            y = (@inferred op(pt))::typeof(pt)
            for name in MPM._fieldnames(MaterialPoint)
                @test getproperty(y, name) == op(getproperty(pt, name))
            end
        end
        # pt + pt, pt - pt
        for op in (+, -)
            y = (@inferred op(pt, pt))::typeof(pt)
            for name in MPM._fieldnames(MaterialPoint)
                @test getproperty(y, name) == op(getproperty(pt, name), getproperty(pt, name))
            end
        end
        # 2p
        y = (@inferred 2pt)::typeof(pt)
        for name in MPM._fieldnames(MaterialPoint)
            @test getproperty(y, name) == 2getproperty(pt, name)
        end
        # pt*2, pt/2
        for op in (*, /)
            y = (@inferred op(pt, 2))::typeof(pt)
            for name in MPM._fieldnames(MaterialPoint)
                @test getproperty(y, name) == op(getproperty(pt, name), 2)
            end
        end
    end
end
