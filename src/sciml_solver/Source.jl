abstract type AbstractSource end



function _ricker(f, t)
    return (1.0 - 2.0 * (π * f * t)^2) * exp(-(π * f * t)^2)
end

struct RickerSource <: AbstractSource
    freq
    tc
    grid
    coefs
    tspan

    function RickerSource(freq, tc, nx, nz; thresh=1e-6, device=CPU())
        t1 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc - 1.0/freq)
        t2 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc + 1.0/freq)

        grid = CartesianIndices((nx,nz))
        coefs = KA.ones(device, 1,1)

        return new(freq,tc,grid,coefs,(t1,t2))
    end
end

@kernel function inject_source_kernel!(du, val, grid, coefs)
    I = @index(Global, Cartesian)
    g = grid[I]
    c = coefs[I]
    du[g] += c * val
end

function (s::RickerSource)(du, t)
    if s.tspan[1] < t < s.tspan[2] # Avoid calculating the source for all t
        τ = t - s.tc
        val = _ricker(s.freq, τ)
        
        # Debug output
        @debug "Injecting source: t=$t, τ=$τ, val=$val, pos=($(s.nx), $(s.nz))"
        
        # Use kernel to avoid scalar indexing issues with CUDA
        device = get_backend(du)
        kernel! = inject_source_kernel!(device)
        kernel!(du, val, s.grid, s.coefs; ndrange=size(s.coefs))
        KA.synchronize(device)
        
    end
    return nothing
end