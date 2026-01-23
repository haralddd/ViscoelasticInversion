using ViscoelasticInversion
using CUDA

if abspath(PROGRAM_FILE) == @__FILE__
    Nx = 10
    Nz = 10
    A = rand(1:9, Nx, Nz)
    
    fdm = Stencil(4, 0)
    pbc = PeriodicBC(fdm, Nx, Nz)
    zbc = ZeroBC(fdm, Nx, Nz)

    println("Periodic BC test:")
    pbc(A)
    display(A)

    println("Zero BC test:")
    zbc(A)
    display(A)


    println("Testing CUDA array")
    Acu = cu(A)
    fdm_gpu = Stencil(4, 0, device=get_backend(Acu))
    println("Periodic BC test:")
    pbc(Acu)

    println("Zero BC test:")
    zbc(Acu)



end