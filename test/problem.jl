@testset "Problem" begin
    for T in (Float32, Float64)
        L = T(1)
        ρ = T(1)         # density
        E = T(4π^2)      # elastic modulus
        m = T(1)         # mass
        v0 = Vec(T(0.1)) # velocity
        V0 = T(1)        # volume
        grid = generategrid([0 L], 1)
        particles = generateparticles([0 L], 1) do x
            Particle(x = x, m = m, V₀ = V0, v = v0)
        end
        prob = Problem(grid, particles)
        push!(prob, BoundaryVelocity(node -> zero(node.v), grid[1]))
        c = √(E/ρ) # elastic wave speed
        dt = T(0.05 * L / c) # 5 percent of CFL condition
        times = 0:dt:1
        for t in times
            MPM.update_stress_first!(prob, step(times)) do p, Δt
                dϵ = Δt * symmetric(p.L)
                p.σ = p.σ + E * dϵ
            end
            @test particles[1].v ≈ v0*cos(√(E/ρ) / L * t) atol = L*0.01
        end
    end
end
