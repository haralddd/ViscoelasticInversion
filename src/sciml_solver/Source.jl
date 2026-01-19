using Roots
using KernelAbstractions

function _ricker(f, t)
    return (1.0 - 2.0 * (π * f * t)^2) * exp(-(π * f * t)^2)
end

struct RickerSource
    freq
    tc
    nx
    nz
    tspan

    function RickerSource(freq, tc, nx, nz; thresh=1e-6)
        t1 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc - 2.0/freq)
        t2 = find_zero(t -> abs(_ricker(freq, t - tc)) - thresh, tc + 2.0/freq)
        return new(freq,tc,nx,nz,(t1,t2))
    end
end

@kernel function inject_source_kernel!(du, val, nx, nz)
    du[nx, nz] += val
end

function (s::RickerSource)(du, t)
    if s.tspan[1] < t < s.tspan[2] # Avoid calculating the source for all t
        τ = t - s.tc
        val = _ricker(s.freq, τ)
        
        device = get_backend(du)
        kernel! = inject_source_kernel!(device)
        kernel!(du, val, s.nx, s.nz; ndrange=(1,1))
    end
    return nothing
end