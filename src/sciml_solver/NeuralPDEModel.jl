#=
Use NeuralPDE together with Flux to define the model and optimization problem symbolically.


=#
using ModelingToolkit, NeuralPDE, Lux, Optimization, OptimizationOptimJL, LineSearches, Plots

@parameters x t
@variables u(..)

ρ(x) = 1.0/5.0
μ(x) = 5.0

Dtt = Differential(t)^2
Dx = Differential(x)

eq = ρ(x)*Dtt(u(x,t)) - Dx(μ(x)*Dx(u(x,t))) ~ 0

ℓ = 15

init_packet(x) = -2x / ℓ^2 * exp(-x^2 / ℓ^2)
bcs = [
    u(x,0) ~ init_packet(x),
    u(-1000,t) ~ 0,
    u(1000,t) ~ 0
]

domains = [
    t ∈ (0.0, 150.0),
    x ∈ (-1000.0, 1000.0)
]

dim = 2
chain = Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))
discretization = PhysicsInformedNN(chain, QuadratureTraining(; batch = 200, abstol = 1e-6, reltol = 1e-6))

@named elasticwavesys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

prob = discretize(elasticwavesys, discretization)

callback = function (p, l)
    println("Loss: $l")
    return false
end

# Optimizer
opt = LBFGS(linesearch=BackTracking())
res = solve(prob, opt, maxiters = 1000)
phi = discretization.phi

vel = √(μ(0)/ρ(0))
true_sol(x,t) = 0.5 * (init_packet(x-vel*t) + init_packet(x+vel*t))

xs = -1000.0:1.0:1000.0

t1 = 0.0
u_predict = [first(phi([x, t1], res.u)) for x in xs]
u_real = [true_sol(x, t1) for x in xs]

plot(xs, u_predict)
plot!(xs, u_real)

# Horrible performance