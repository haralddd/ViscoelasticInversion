struct RickerSource
    freq
    t0
    nx
    nz
end

function (s::RickerSource)(du, t)
    Δt = t - s.t0
    du[s.nx, s.nz] += (1.0 - 2.0 * (π * s.freq * Δt)^2) * exp(-(π * s.freq * Δt)^2)
    return nothing
end