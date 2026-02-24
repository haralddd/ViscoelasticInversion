"""
Example: Energy Natural Gradient for Viscoelastic FWI

This example shows how to use the Energy Natural Gradient optimizer
for Full Waveform Inversion with:
1. Neural network parameterization (exact Gram matrix)
2. Gridded parameterization (stochastic approximation)
"""

using ViscoelasticInversion
using LinearAlgebra
using ForwardDiff

#=============================================================================
# Example 1: Neural Network Parameterization
# - Small number of parameters (network weights)
# - Exact Gram matrix computation
=============================================================================#

"""
Simple MLP that outputs velocity field from coordinates.
Vp(x, z) = MLP([x, z]; θ)
"""
struct SimpleVelocityNN{T}
    weights::Vector{T}
    layer_sizes::Vector{Int}
end

function SimpleVelocityNN(layer_sizes::Vector{Int}; T=Float64)
    # Count total parameters
    n_params = 0
    for i in 1:length(layer_sizes)-1
        n_params += layer_sizes[i] * layer_sizes[i+1]  # weights
        n_params += layer_sizes[i+1]                    # biases
    end
    weights = randn(T, n_params) .* 0.1
    return SimpleVelocityNN(weights, layer_sizes)
end

function (nn::SimpleVelocityNN)(x::T, z::T) where T
    # Forward pass through MLP
    input = [x, z]
    idx = 1
    h = input
    
    for i in 1:(length(nn.layer_sizes)-1)
        n_in = nn.layer_sizes[i]
        n_out = nn.layer_sizes[i+1]
        
        # Extract weights and biases
        W = reshape(nn.weights[idx:idx+n_in*n_out-1], n_out, n_in)
        idx += n_in * n_out
        b = nn.weights[idx:idx+n_out-1]
        idx += n_out
        
        # Linear + activation (tanh for hidden, identity for output)
        h = W * h .+ b
        if i < length(nn.layer_sizes) - 1
            h = tanh.(h)
        end
    end
    
    return h[1]  # Single output: velocity
end

function nn_to_velocity_field(nn::SimpleVelocityNN{T}, Nx::Int, Nz::Int, dx, dz) where T
    # Use eltype of weights to support ForwardDiff Dual numbers
    Vp = zeros(T, Nx, Nz)
    for j in 1:Nz
        for i in 1:Nx
            x = T((i - 1) * dx)
            z = T((j - 1) * dz)
            Vp[i, j] = nn(x / 1000, z / 1000)  # Normalize coords
        end
    end
    return Vp
end

"""
Example: NN-based FWI with Energy Natural Gradient

This demonstrates the workflow but uses a simplified forward model.
Replace with actual viscoelastic solver for real applications.
"""
function example_nn_fwi()
    println("="^60)
    println("Example 1: Neural Network Parameterization")
    println("="^60)
    
    # Setup
    Nx, Nz = 50, 50
    dx, dz = 20.0, 20.0
    
    # True velocity model (target)
    Vp_true = zeros(Nx, Nz)
    for j in 1:Nz, i in 1:Nx
        Vp_true[i, j] = 2000.0 + 500.0 * sin(π * i / Nx) * sin(π * j / Nz)
    end
    
    # Neural network parameterization
    # Input: (x, z), Hidden: 16, 16, Output: Vp
    nn = SimpleVelocityNN([2, 16, 16, 1])
    θ₀ = copy(nn.weights)
    p = length(θ₀)
    println("Number of parameters: $p")
    
    # Loss function: ||Vp(θ) - Vp_true||²
    function loss_fn(θ)
        nn_trial = SimpleVelocityNN(θ, nn.layer_sizes)
        Vp = nn_to_velocity_field(nn_trial, Nx, Nz, dx, dz)
        return sum((Vp .- Vp_true).^2) / (Nx * Nz)
    end
    
    # Gradient via AD
    function grad_fn(θ)
        return ForwardDiff.gradient(loss_fn, θ)
    end
    
    # Sensitivity function for exact Gram
    # Returns [∂θ_1 Vp, ∂θ_2 Vp, ...] as flattened vectors
    function sensitivity_fn(θ)
        sensitivities = Vector{Vector{Float64}}(undef, p)
        
        for i in 1:p
            # Compute ∂Vp/∂θ_i via forward-mode AD
            v = zeros(p)
            v[i] = 1.0
            
            sens = ForwardDiff.derivative(t -> begin
                nn_trial = SimpleVelocityNN(θ .+ t .* v, nn.layer_sizes)
                vec(nn_to_velocity_field(nn_trial, Nx, Nz, dx, dz))
            end, 0.0)
            
            sensitivities[i] = sens
        end
        
        return sensitivities
    end
    
    # Initialize optimizer and state
    opt = EnergyNaturalGradientOptimizer(
        n_probes=0,                    # Exact Gram (small p)
        regularization=1e-6,
        line_search_points=15,
        line_search_bounds=(1e-8, 1.0),
        stochastic_threshold=1000
    )
    
    state = OptimizationState(θ₀)
    
    # Run optimization
    println("\nInitial loss: $(loss_fn(θ₀))")
    println("Running Energy Natural Gradient optimization...")
    
    optimize!(state, opt, loss_fn, grad_fn, sensitivity_fn, 50; verbose=true)
    
    println("\nFinal loss: $(state.loss_history[end])")
    println("Iterations: $(state.iteration)")
    
    return state
end

#=============================================================================
# Example 2: Gridded Parameterization (Stochastic Gram)
# - Large number of parameters (grid cells)
# - Stochastic approximation of Gram matrix
=============================================================================#

"""
Example: Gridded FWI with Stochastic Energy Natural Gradient

For large parameter counts, use randomized Gram matrix estimation.
"""
function example_gridded_fwi()
    println("\n" * "="^60)
    println("Example 2: Gridded Parameterization (Stochastic)")
    println("="^60)
    
    # Small grid for demonstration
    Nx, Nz = 20, 20
    p = Nx * Nz
    println("Number of parameters: $p")
    
    # True velocity model
    Vp_true = zeros(Nx, Nz)
    for j in 1:Nz, i in 1:Nx
        Vp_true[i, j] = 2000.0 + 300.0 * exp(-((i-Nx/2)^2 + (j-Nz/2)^2) / 50)
    end
    
    # Initial guess
    θ₀ = fill(2000.0, p)
    
    # Loss function
    function loss_fn(θ)
        Vp = reshape(θ, Nx, Nz)
        return sum((Vp .- Vp_true).^2) / p
    end
    
    # Gradient
    function grad_fn(θ)
        Vp = reshape(θ, Nx, Nz)
        return vec(2.0 .* (Vp .- Vp_true) ./ p)
    end
    
    # JVP function for stochastic Gram
    # In real FWI: this would involve forward sensitivity wavefield
    function sensitivity_fn(θ)
        # Return a JVP function
        return function jvp(v)
            # For this simple quadratic loss, J = I
            # So JVP(v) = v (identity mapping)
            return v
        end
    end
    
    # Initialize optimizer with stochastic settings
    opt = EnergyNaturalGradientOptimizer(
        n_probes=20,                   # Stochastic Gram with 20 probes
        regularization=1e-4,
        line_search_points=10,
        line_search_bounds=(1e-6, 0.5),
        stochastic_threshold=100       # Use stochastic if p > 100
    )
    
    state = OptimizationState(θ₀)
    
    # Run optimization
    println("\nInitial loss: $(loss_fn(θ₀))")
    println("Running Stochastic Energy Natural Gradient optimization...")
    
    optimize!(state, opt, loss_fn, grad_fn, sensitivity_fn, 30; verbose=true)
    
    println("\nFinal loss: $(state.loss_history[end])")
    println("Iterations: $(state.iteration)")
    
    return state
end

#=============================================================================
# Example 3: Integration with ViscoelasticProblem
# - Shows how to connect to the actual wave solver
=============================================================================#

"""
Sketch of how to integrate with the full viscoelastic solver.

NOTE: This is a template - actual implementation requires:
1. Setting up observed data
2. Implementing adjoint-state gradient
3. Computing wavefield sensitivities
"""
function example_viscoelastic_integration_sketch()
    println("\n" * "="^60)
    println("Example 3: Viscoelastic Integration (Sketch)")
    println("="^60)
    
    println("""
    
    To integrate with the full viscoelastic solver:
    
    1. LOSS FUNCTION:
       ```julia
       function loss_fn(θ)
           # Update model with new parameters
           model = create_model_from_params(θ)
           params = Parameters(model, ...)
           prob = make_problem(params; tspan=(0.0, 2.0))
           sol = solve_problem(prob)
           
           # Extract synthetic data at receiver locations
           d_syn = extract_receivers(sol)
           
           # Return misfit
           return sum((d_syn .- d_obs).^2)
       end
       ```
    
    2. GRADIENT (Adjoint-State):
       ```julia
       function grad_fn(θ)
           # Forward solve
           sol_fwd = solve_forward(θ)
           
           # Compute adjoint source
           adj_src = d_syn - d_obs
           
           # Backward (adjoint) solve
           sol_adj = solve_adjoint(adj_src)
           
           # Compute gradient via zero-lag correlation
           return compute_gradient(sol_fwd, sol_adj)
       end
       ```
    
    3. SENSITIVITIES (For Natural Gradient):
       For NN parameterization:
       ```julia
       function sensitivity_fn(θ)
           return compute_sensitivities_forwardmode(forward_solve, θ)
       end
       ```
       
       For gridded parameters:
       ```julia
       function sensitivity_fn(θ)
           return make_jvp_function(forward_solve, θ)
       end
       ```
    
    The Energy Natural Gradient is most beneficial when:
    - Using neural network parameterizations
    - The loss landscape is ill-conditioned
    - Standard gradient descent converges slowly
    """)
end

# Run examples
if abspath(PROGRAM_FILE) == @__FILE__
    example_nn_fwi()
    example_gridded_fwi()
    example_viscoelastic_integration_sketch()
end
