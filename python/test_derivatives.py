import numpy as np
import torch
import logging
from SpatialScheme import FD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_derivatives():
    logger.info("Testing derivative accuracy...")
    
    # Grid parameters
    Nx, Nz = 100, 100
    Lx, Lz = 2 * np.pi, 2 * np.pi
    dx, dz = Lx / Nx, Lz / Nz
    
    # Create coordinate grids
    x = np.linspace(0, Lx, Nx, dtype=np.float32)
    z = np.linspace(0, Lz, Nz, dtype=np.float32)
    X, Z = np.meshgrid(x, z, indexing='ij')
    
    # Test function: u(x,z) = sin(x) * cos(z)
    u = np.sin(X) * np.cos(Z)
    
    # Analytical derivatives
    dudx_analytical = np.cos(X) * np.cos(Z)
    dudz_analytical = -np.sin(X) * np.sin(Z)
    
    # Convert to torch and add batch/channel dimensions
    u_torch = torch.from_numpy(u).reshape(1, 1, Nx, Nz)
    
    # Create finite difference scheme
    order = 8
    fd_cpu = FD(order, dx, dz, device=torch.device('cpu'))
    
    # Compute numerical derivatives
    dudx_numerical = fd_cpu.ddx(u_torch).squeeze().numpy()
    dudz_numerical = fd_cpu.ddz(u_torch).squeeze().numpy()
    
    # Compute errors (excluding boundary points affected by padding)
    pad = order // 2
    interior = slice(pad, Nx - pad), slice(pad, Nz - pad)
    
    err_x = dudx_numerical[interior] - dudx_analytical[interior]
    err_z = dudz_numerical[interior] - dudz_analytical[interior]
    
    max_err_x = np.max(np.abs(err_x))
    max_err_z = np.max(np.abs(err_z))
    rms_err_x = np.sqrt(np.mean(err_x**2))
    rms_err_z = np.sqrt(np.mean(err_z**2))
    
    logger.info(f"\nResults (order={order}, dx={dx:.6f}, dz={dz:.6f}):")
    logger.info(f"  ∂u/∂x - Max error: {max_err_x:.2e}, RMS error: {rms_err_x:.2e}")
    logger.info(f"  ∂u/∂z - Max error: {max_err_z:.2e}, RMS error: {rms_err_z:.2e}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        logger.info("\nTesting GPU...")
        u_gpu = u_torch.to(torch.device('cuda'))
        fd_gpu = FD(order, dx, dz, device=torch.device('cuda'))
        
        dudx_gpu = fd_gpu.ddx(u_gpu).cpu().squeeze().numpy()
        dudz_gpu = fd_gpu.ddz(u_gpu).cpu().squeeze().numpy()
        
        err_gpu_x = np.max(np.abs(dudx_gpu[interior] - dudx_numerical[interior]))
        err_gpu_z = np.max(np.abs(dudz_gpu[interior] - dudz_numerical[interior]))
        
        logger.info(f"  GPU vs CPU - Max diff ∂u/∂x: {err_gpu_x:.2e}")
        logger.info(f"  GPU vs CPU - Max diff ∂u/∂z: {err_gpu_z:.2e}")
    
    return max_err_x, max_err_z, rms_err_x, rms_err_z

if __name__ == "__main__":
    test_derivatives()
