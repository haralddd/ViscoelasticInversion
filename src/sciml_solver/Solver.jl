struct Solver



    # Pre-allocated steps
    dxvx
    dzvx

    dxvz
    dzvz

    dx_sxx
    dx_szx # = dz_sxz, but ddx is faster than ddz
    dz_szz
end