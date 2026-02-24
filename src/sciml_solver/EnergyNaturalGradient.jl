"""
Energy Natural Gradient optimizer for FWI.

Implements Algorithm 1 from the PINN natural gradient literature:
- Computes Gram matrix G_E based on energy inner product
- Solves least-squares for natural gradient direction
- Line search over logarithmic grid

Supports both:
- Small parameter sets (neural networks): exact Gram matrix
- Large parameter sets (gridded): stochastic approximation
"""

using LinearAlgebra
using ForwardDiff

abstract type AbstractEnergyNaturalGradient end

#=============================================================================
# Energy inner product definitions
=============================================================================#

"""
    EnergyInnerProduct

Defines the energy inner product D²E(u, v) for wavefields.
For viscoelastic waves: E = ∫ (ρ v·v + s:C⁻¹:s) dx
"""
struct EnergyInnerProduct{T}
    ρ::T      # density field
    C_inv::T  # inverse stiffness (for stress energy)
end

"""
Simple L² inner product (default, works for most cases).
"""
function energy_inner_product(u1, u2, ::Nothing)
    return dot(vec(u1), vec(u2))
end

"""
Weighted energy inner product with density.
"""
function energy_inner_product(u1, u2, eip::EnergyInnerProduct)
    # Weighted L² with density as weight
    return sum(eip.ρ .* u1 .* u2)
end

#=============================================================================
# Gram matrix computation
=============================================================================#

"""
    compute_gram_exact(sensitivities, energy_ip)

Compute exact Gram matrix G_E[i,j] = D²E(∂θ_i u, ∂θ_j u).
For small parameter counts (neural networks).

# Arguments
- `sensitivities`: Vector of wavefield sensitivities [∂θ_1 u, ∂θ_2 u, ...]
- `energy_ip`: Energy inner product definition (or nothing for L²)
"""
function compute_gram_exact(sensitivities::Vector, energy_ip=nothing)
    p = length(sensitivities)
    T = eltype(first(sensitivities))
    G = zeros(T, p, p)
    
    @inbounds for j in 1:p
        for i in j:p  # Exploit symmetry
            G[i, j] = energy_inner_product(sensitivities[i], sensitivities[j], energy_ip)
            if i != j
                G[j, i] = G[i, j]
            end
        end
    end
    
    return G
end

"""
    compute_gram_stochastic(jvp_fn, grad, n_probes, energy_ip)

Approximate Gram matrix using randomized probing (for large parameter sets).
Uses Hutchinson trace estimator.

# Arguments
- `jvp_fn`: Function θ -> JVP(v) that computes Jacobian-vector products
- `p`: Number of parameters
- `n_probes`: Number of random probe vectors
- `energy_ip`: Energy inner product definition
"""
function compute_gram_stochastic(jvp_fn, p::Int, n_probes::Int, energy_ip=nothing)
    T = Float64  # Could be parameterized
    
    # Accumulate outer products from random probes
    G_approx = zeros(T, p, p)
    
    for _ in 1:n_probes
        # Random Rademacher vector
        z = rand([-one(T), one(T)], p)
        
        # Compute J*z (wavefield sensitivity in direction z)
        Jz = jvp_fn(z)
        
        # Compute Gˢᵀz via adjoint (VJP)
        # For symmetric G: G*z ≈ Jᵀ * D²E * J * z
        # This is a stochastic approximation
        G_approx .+= (z * z') .* energy_inner_product(Jz, Jz, energy_ip)
    end
    
    return G_approx ./ n_probes
end

#=============================================================================
# Natural gradient computation
=============================================================================#

"""
    compute_natural_gradient(G, grad; regularization=1e-8)

Solve least-squares problem: ∇ᴱL = argmin_ψ ||G_E ψ - ∇L||²

# Arguments
- `G`: Gram matrix
- `grad`: Standard gradient ∇L(θ)
- `regularization`: Tikhonov regularization for stability
"""
function compute_natural_gradient(G::AbstractMatrix, grad::AbstractVector; regularization=1e-8)
    p = size(G, 1)
    
    # Add regularization for numerical stability
    G_reg = G + regularization * I(p)
    
    # Solve least squares: G_reg * ψ = grad
    # Using backslash (will use appropriate factorization)
    return G_reg \ grad
end

#=============================================================================
# Line search
=============================================================================#

"""
    line_search_log(loss_fn, θ, direction; n_points=10, bounds=(1e-6, 1.0))

Grid search over logarithmically spaced learning rates.

# Arguments
- `loss_fn`: Function θ -> L(θ)
- `θ`: Current parameters
- `direction`: Search direction (natural gradient)
- `n_points`: Number of grid points
- `bounds`: (min_η, max_η) bounds for learning rate
"""
function line_search_log(loss_fn, θ::AbstractVector, direction::AbstractVector; 
                         n_points::Int=10, bounds::Tuple=(1e-6, 1.0))
    η_min, η_max = bounds
    
    # Logarithmically spaced grid
    log_ηs = range(log10(η_min), log10(η_max), length=n_points)
    ηs = 10 .^ log_ηs
    
    best_η = ηs[1]
    best_loss = Inf
    
    for η in ηs
        θ_trial = θ .- η .* direction
        loss_trial = loss_fn(θ_trial)
        
        if loss_trial < best_loss
            best_loss = loss_trial
            best_η = η
        end
    end
    
    return best_η, best_loss
end

#=============================================================================
# Main optimizer struct and step function
=============================================================================#

"""
    EnergyNaturalGradientOptimizer

Energy Natural Gradient optimizer with line search.

# Fields
- `n_probes`: Number of probes for stochastic Gram (0 = exact)
- `regularization`: Tikhonov regularization
- `line_search_points`: Grid points for line search
- `energy_ip`: Energy inner product definition
- `stochastic_threshold`: Use stochastic Gram if p > this
"""
Base.@kwdef struct EnergyNaturalGradientOptimizer{T}
    n_probes::Int = 50
    regularization::Float64 = 1e-8
    line_search_points::Int = 10
    line_search_bounds::Tuple{Float64, Float64} = (1e-6, 1.0)
    energy_ip::T = nothing
    stochastic_threshold::Int = 1000
end

"""
    OptimizationState

Mutable state for tracking optimization progress.
"""
mutable struct OptimizationState{T}
    θ::Vector{T}
    iteration::Int
    loss_history::Vector{T}
    η_history::Vector{T}
end

function OptimizationState(θ₀::AbstractVector{T}) where T
    return OptimizationState(
        copy(vec(θ₀)),
        0,
        T[],
        T[]
    )
end

"""
    step!(state, opt, loss_fn, grad_fn, sensitivity_fn)

Perform one Energy Natural Gradient step.

# Arguments
- `state`: OptimizationState
- `opt`: EnergyNaturalGradientOptimizer
- `loss_fn`: θ -> L(θ)
- `grad_fn`: θ -> ∇L(θ)
- `sensitivity_fn`: Either:
  - θ -> [∂θ_1 u, ∂θ_2 u, ...] for exact Gram
  - θ -> jvp_function for stochastic Gram
"""
function step!(state::OptimizationState, opt::EnergyNaturalGradientOptimizer,
               loss_fn, grad_fn, sensitivity_fn)
    
    θ = state.θ
    p = length(θ)
    
    # 1. Compute standard gradient
    grad = grad_fn(θ)
    
    # 2. Build Gram matrix
    if p <= opt.stochastic_threshold && opt.n_probes == 0
        # Exact Gram matrix (small parameter set)
        sensitivities = sensitivity_fn(θ)
        G = compute_gram_exact(sensitivities, opt.energy_ip)
    else
        # Stochastic approximation (large parameter set)
        jvp_fn = sensitivity_fn(θ)
        G = compute_gram_stochastic(jvp_fn, p, opt.n_probes, opt.energy_ip)
    end
    
    # 3. Compute natural gradient
    nat_grad = compute_natural_gradient(G, grad; regularization=opt.regularization)
    
    # 4. Line search
    η_opt, loss_new = line_search_log(
        loss_fn, θ, nat_grad;
        n_points=opt.line_search_points,
        bounds=opt.line_search_bounds
    )
    
    # 5. Update parameters
    @. state.θ = θ - η_opt * nat_grad
    state.iteration += 1
    push!(state.loss_history, loss_new)
    push!(state.η_history, η_opt)
    
    return loss_new, η_opt
end

"""
    optimize!(state, opt, loss_fn, grad_fn, sensitivity_fn, n_iterations; 
              callback=nothing, verbose=true)

Run Energy Natural Gradient optimization.

# Arguments
- `state`: OptimizationState (modified in place)
- `opt`: EnergyNaturalGradientOptimizer
- `loss_fn`: θ -> L(θ)
- `grad_fn`: θ -> ∇L(θ)
- `sensitivity_fn`: Sensitivity/JVP function
- `n_iterations`: Maximum iterations
- `callback`: Optional function called each iteration
- `verbose`: Print progress
"""
function optimize!(state::OptimizationState, opt::EnergyNaturalGradientOptimizer,
                   loss_fn, grad_fn, sensitivity_fn, n_iterations::Int;
                   callback=nothing, verbose::Bool=true)
    
    for k in 1:n_iterations
        loss, η = step!(state, opt, loss_fn, grad_fn, sensitivity_fn)
        
        if verbose && (k % 10 == 0 || k == 1)
            @info "Iteration $k: loss = $(round(loss; sigdigits=6)), η = $(round(η; sigdigits=3))"
        end
        
        if callback !== nothing
            callback(state, k, loss, η)
        end
        
        # Check for convergence (simple criterion)
        if length(state.loss_history) > 1
            rel_change = abs(state.loss_history[end] - state.loss_history[end-1]) / 
                         (abs(state.loss_history[end-1]) + 1e-12)
            if rel_change < 1e-10
                @info "Converged at iteration $k (relative change: $rel_change)"
                break
            end
        end
    end
    
    return state
end

#=============================================================================
# Helper: compute sensitivities via forward-mode AD
=============================================================================#

"""
    compute_sensitivities_forwardmode(forward_solve, θ)

Compute wavefield sensitivities ∂θ_i u using forward-mode AD.
For small parameter sets (neural networks).

# Arguments
- `forward_solve`: Function θ -> u(θ) that returns wavefield
- `θ`: Parameter vector
"""
function compute_sensitivities_forwardmode(forward_solve, θ::AbstractVector)
    p = length(θ)
    
    # Compute Jacobian columns via forward-mode AD
    sensitivities = Vector{Any}(undef, p)
    
    for i in 1:p
        # Unit vector in direction i
        v = zeros(eltype(θ), p)
        v[i] = one(eltype(θ))
        
        # Forward-mode directional derivative
        sensitivities[i] = ForwardDiff.derivative(
            t -> forward_solve(θ .+ t .* v),
            zero(eltype(θ))
        )
    end
    
    return sensitivities
end

"""
    make_jvp_function(forward_solve, θ)

Create a JVP function for stochastic Gram computation.

# Arguments  
- `forward_solve`: Function θ -> u(θ)
- `θ`: Current parameters
"""
function make_jvp_function(forward_solve, θ::AbstractVector)
    return function jvp(v::AbstractVector)
        # JVP via forward-mode AD
        return ForwardDiff.derivative(
            t -> forward_solve(θ .+ t .* v),
            zero(eltype(θ))
        )
    end
end
