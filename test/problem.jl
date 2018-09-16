@testset "Problem" begin
    @testset "Vibration of a single material point" begin
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
            prob = Problem(grid, particles; gravity = 0)
            push!(prob, BoundaryVelocity(node -> zero(node.v), grid[1]))
            c = √(E/ρ) # elastic wave speed
            dt = T(0.1 * L / c) # 5 percent of CFL condition
            t = T(0)
            for _ in 1:100
                MPM.update_stress_first!(prob, dt) do p, Δt
                    dϵ = Δt * symmetric(p.L)
                    p.σ = p.σ + E * dϵ
                end
                t += dt
                @test particles[1].v ≈ v0*cos(√(E/ρ) / L * t) atol = L*0.01
            end
        end
    end
    @testset "Free fall of a single material point" begin
        for T in (Float32, Float64)
            H = T(10)
            E = T(1)            # elastic modulus
            m = T(1)            # mass
            v0 = zero(Vec{2,T}) # velocity
            V0 = T(1)           # volume
            g = T(9.81)
            grid = generategrid([0 H/10; 0 H], 1, 10)
            particles = generateparticles([0 H/10; 9H/10 H], 1, 1) do x
                Particle(x = x, m = m, V₀ = V0, v = v0)
            end
            prob = Problem(grid, particles; gravity = g)
            dt = T(0.1)
            t = T(0)
            for _ in 1:10
                MPM.update_stress_first!(prob, dt) do p, Δt
                    dϵ = Δt * symmetric(p.L)
                    p.σ = p.σ + E * dϵ
                end
                t += dt
                @test particles[1].v[2] ≈ -g*t
            end
        end
    end
end
