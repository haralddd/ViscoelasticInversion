module ViscoelasticInversion

using Roots
using KernelAbstractions
using OrdinaryDiffEq
using DiffEqCallbacks

KA = KernelAbstractions
import Base./ # For joinpath(string...) convenience

include("common/utils.jl")
include("common/Stencil.jl")

include("sciml_solver/Preallocated.jl")
include("sciml_solver/Source.jl")
include("sciml_solver/BoundaryConditions.jl")
include("sciml_solver/Models.jl")
include("sciml_solver/Parameters.jl")
include("sciml_solver/ViscoelasticProblem.jl")

export Stencil
export Preallocated
export RickerSource
export AbstractBC, DirichletBC, NeumannBC, PeriodicBC, AbsorbingBC
export Parameters
export AbstractModel, IsotropicModel, VTIModel, TTIModel
export make_problem, solve_problem

end # module