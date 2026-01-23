
# Preferred device storage types in the stencil struct
preferred_float(::CPU) = Float64
preferred_float(::GPU) = Float32
preferred_int(::CPU) = Int
preferred_int(::GPU) = Int

/(parts::AbstractString...) = joinpath(parts)