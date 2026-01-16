abstract type PointSource end

struct Ricker <: PointSource
    fM!
    t0
end

function get_value(s::T, t) where {T <: PointSource}
    error("get_value for `$(typeof(s)) <: PointSource` not implemented")
end
function get_value(s::Ricker, t)
    Δt = t - s.t0
    return (1.0 - 2.0 * (π * s.fM * Δt)^2) * exp(-(π * s.fM * Δt)^2)
end


sinc(100π)


function source_interpolation(f,x,x0)
    f0 = s.f0
    t0 = s.t0
    return sinc(x-x0)
end