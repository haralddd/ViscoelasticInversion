abstract type AbstractModel end

@kwdef struct IsotropicModel <: AbstractModel
    C11
    C13
    C33
    C55

    "Constructs homogeneous isotropic model from Lamé parameters"
    function IsotropicModel(λ::T, μ::T, Nx, Nz) where T <: Real
        C11 = C33 = fill(λ + 2μ, Nx, Nz)
        C13 = fill(λ, Nx, Nz)
        C55 = fill(μ, Nx, Nz)
        return new(C11, C13, C33, C55)
    end
end

@kwdef struct VTIModel <: AbstractModel
    C11
    C13
    C33
    C55
end

@kwdef struct TTIModel <: AbstractModel
    C11
    C13
    C15
    C33
    C35
    C55
end

function size(m::T) where T <: AbstractModel
    return size(m.C11)
end

