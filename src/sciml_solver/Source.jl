using Roots
using KernelAbstractions

function _ricker(f, t)
    return (1.0 - 2.0 * (π * f * t)^2) * exp(-(π * f * t)^2)
end

struct RickerSource
    freq
    tc
    grid
    coefs
    tspan

    function RickerSource(freq, tc, xs, x0; width=0, thresh=1e-6)
        t1 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc - 1.0/freq)
        t2 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc + 1.0/freq)
        @assert isodd(width) ""

        search

        grid = CartesianIndices((-width:+width, -width:width)) .+ CartesianIndex(nx, nz)

        # FIXME: Correct interpolation of source
        coefs = ones()

        coefs ./= sum(coefs)

        return new(freq,tc,nx,nz,(t1,t2), width)
    end
end

@kernel function inject_source_kernel!(du, val, grid, coefs)
    I = @index(Global, Cartesian)
    g = grid[I]
    c = coefs[I]
    du[g] += val
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
        kernel!(du, val, s.nx, s.nz; ndrange=(width,width))
        KA.synchronize(device)
        
        # Verify injection
        @debug begin
            injected_val = du[s.nx, s.nz]
            "Injected value at ($(s.nx), $(s.nz)): $injected_val"
        end
    end
    return nothing
end