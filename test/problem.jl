@testset "Problem" begin
    @testset "Vibration of a single material point" begin
        for T in (Float32, Float64)
            L = 1
            ρ = 1            # density
            E = 4π^2         # elastic modulus
            c = √(E/ρ)       # elastic wave speed
            dt = 0.1 * L / c # 5 percent of CFL condition
            grid = @inferred generategrid(T[0 L], 1)
            particles = @inferred generateparticles(x -> Particle(x = x, m = 1, V₀ = 1, v = Vec(0.1)), T[0 L], 1)
            prob = Problem(grid, (0, 100dt)) do p, Δt
                dϵ = Δt * symmetric(p.L)
                p.σ = p.σ + E * dϵ
            end
            push!(prob, BoundaryVelocity(node -> zero(node.v), grid[1]))
            sol = solve(prob, particles, USF(), dt = dt)
            for s in sol
                v0 = sol[1].particles[1].v
                @test s.particles[1].v::Vec{1, T} ≈ v0*cos(√(E/ρ) / L * s.t) atol = L*0.01
            end
        end
    end
    @testset "Free fall of a single material point" begin
        for T in (Float32, Float64)
            H = 10
            E = 1 # elastic modulus
            dt = 0.01
            grid = @inferred generategrid(T[0 H/10; 0 H], 1, 10)
            particles = @inferred generateparticles(x -> Particle(x = x, m = 1, V₀ = 1), T[0 H/10; 9H/10 H], 1, 1)
            prob = Problem(grid, (0, 100dt); gravity = Vec(0, -9.81)) do p, Δt
                dϵ = Δt * symmetric(p.L)
                p.σ = p.σ + E * dϵ
            end
            sol = solve(prob, particles, USF(), dt = dt)
            for s in sol
                g = prob.gravity
                @test s.particles[1].v::Vec{2, T} ≈ g*s.t
            end
        end
    end
end
