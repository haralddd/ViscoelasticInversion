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

    function RickerSource(freq, tc, grid, coefs; thresh=1e-6)
        t1 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc - 1.0/freq)
        t2 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc + 1.0/freq)

        return new(freq, tc, grid, coefs, (t1, t2))
    end

    function RickerSource(freq, tc, nx::Int, nz::Int; thresh=1e-6, device=CPU())

        I = preferred_int(device)
        F = preferred_float(device)

        grid = allocate(device, Tuple{I, I}, (1, 1))
        coefs = allocate(device, F, (1,1))
        copyto!(grid, [(nx, nz)])
        copyto!(coefs, [1.0])

        return RickerSource(freq, tc, grid, coefs, thresh=thresh)
    end
end

@kernel function inject_source_kernel!(du, val, grid, coefs)
    I = @index(Global, Cartesian)
    gx, gz = grid[I]
    c = coefs[I]
    du[gx, gz] += c * val
end

function (s::RickerSource)(du, t)
    if s.tspan[1] < t < s.tspan[2] # Avoid calculating the source for all t
        τ = t - s.tc
        val = _ricker(s.freq, τ)
        
        # Debug output
        @debug "Injecting source: t=$t, τ=$τ, val=$val at grid point $(s.grid)"
        
        # Use kernel to avoid scalar indexing issues with CUDA
        device = get_backend(du)
        kernel! = inject_source_kernel!(device)
        kernel!(du, val, s.grid, s.coefs; ndrange=size(s.coefs))
        KA.synchronize(device)
        
    end
    return nothing
end