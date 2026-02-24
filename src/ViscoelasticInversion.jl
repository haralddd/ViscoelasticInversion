module ViscoelasticInversion

using Roots
using KernelAbstractions
using OrdinaryDiffEq
using DiffEqCallbacks
using RecursiveArrayTools
using Optim
using QuadGK
KA = KernelAbstractions
import Base./ # For joinpath(string...) convenience

include("common/utils.jl")
include("common/Stencil.jl")

include("sciml_solver/Preallocated.jl")
include("sciml_solver/Source.jl")
include("sciml_solver/BoundaryConditions.jl")
include("sciml_solver/CPML.jl")
include("sciml_solver/Models.jl")
include("sciml_solver/Parameters.jl")
include("sciml_solver/ViscoelasticProblem.jl")
include("sciml_solver/EnergyNaturalGradient.jl")

export Stencil
export Preallocated
export RickerSource
export AbstractBC, DirichletBC, NeumannBC, PeriodicBC, AbsorbingBC
export CPMLConfig, CPMLBC, CPMLCoefficients, CPMLMemory, FreeSurfaceBC
export Parameters
export AbstractModel, IsotropicModel, VTIModel, TTIModel
export make_problem, solve_problem
export EnergyNaturalGradientOptimizer, OptimizationState
export step!, optimize!, compute_sensitivities_forwardmode, make_jvp_function

end # module