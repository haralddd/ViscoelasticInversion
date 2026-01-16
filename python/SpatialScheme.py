import time
import logging
import torch
import torch.nn.functional as ptf
import numpy as np

logger = logging.getLogger(__name__)

# ==== Spatial schemes ====
def _fornberg_weights(x, x0 = 0.0, n = 1):
    m = len(x)
    assert m > 1, "Fornberg weight calculation received empty abscissa array"
    weights = torch.zeros((m, n + 1))

    c_1, c_4 = 1, x[0] - x0
    c_6, c_7 = 0.0, 0.0

    weights[0, 0] = 1
    for i in range(1, m):
        j = torch.arange(0, min(i, n) + 1)
        c_2, c_5, c_4 = 1, c_4, x[i] - x0
        for v in range(i):
            c_3 = x[i] - x[v]
            c_2, c_6, c_7 = c_2 * c_3, j * weights[v, j - 1], weights[v, j]
            weights[v, j] = (c_4 * c_7 - c_6) / c_3
        weights[i, j] = c_1 * (c_6 - c_5 * c_7) / c_2
        c_1 = c_2
    
    return weights[:, -1]


def _get_central_coeffs(order: int, dx: float, device: torch.device = torch.device('cpu')):
    assert order % 2 == 0, "Order must be even for central difference"
    
    # Generate stencil points for central difference
    half = order // 2
    x = torch.arange(-half, half + 1) * dx
    
    # Use Fornberg algorithm to get coefficients
    coeffs = _fornberg_weights(x, x0=0.0, n=1)  # 1st derivative
    return coeffs.to(device)


class SpatialScheme:
    def __init__(self, dx: float, dz: float):
        self.dx = dx
        self.dz = dz


class FD(SpatialScheme):
    def __init__(self, order: int, dx: float, dz: float, device: torch.device = None):
        super().__init__(dx, dz)

        # Store coefficients on device (GPU if available)   
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.order = order
        self.cxs_coeffs = _get_central_coeffs(order, dx, self.device)
        self.czs_coeffs = _get_central_coeffs(order, dz, self.device)
        self.cxs = self.cxs_coeffs.reshape(1,1,1,order+1) # Prepare for conv2d (operates on width/x-direction)
        self.czs = self.czs_coeffs.reshape(1,1,order+1,1) # Prepare for conv2d (operates on height/z-direction)

    def ddx(self, field: torch.Tensor) -> torch.Tensor:
        return torch.conv2d(field, self.cxs)

    def ddz(self, field: torch.Tensor) -> torch.Tensor:
        return torch.conv2d(field, self.czs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    N = 4096
    M = 4096
    order = 8
    iters = 1000

    cpu_u = torch.randn((1,1,M, N), dtype=torch.float, device='cpu')
    fd_cpu = FD(order, 1.0, 1.0, torch.device('cpu'))
    
    # Compute derivatives
    time_begin = time.time()
    cxts = np.zeros(iters)
    for i in range(0, iters):
        cpu_u = torch.randn((1,1,M, N), dtype=torch.float, device='cpu')
        time_begin = time.time()
        _ = fd_cpu.ddx(cpu_u)
        cxts[i] = time.time() - time_begin
    
    czts = np.zeros(iters)
    for i in range(0, iters):
        cpu_u = torch.randn((1,1,M, N), dtype=torch.float, device='cpu')
        time_begin = time.time()
        _ = fd_cpu.ddz(cpu_u)
        czts[i] = time.time() - time_begin


    gpu_u = torch.randn((1,1,M, N), dtype=torch.float, device='cuda')
    fd_gpu = FD(order, 1.0, 1.0, torch.device('cuda'))

    gxts = np.zeros(iters)
    for i in range(0, iters):
        gpu_u = torch.randn((1,1,M, N), dtype=torch.float, device='cuda')
        time_begin = time.time()
        _ = fd_gpu.ddx(gpu_u)
        torch.cuda.synchronize()  # CRITICAL: Wait for GPU to finish
        gxts[i] = time.time() - time_begin
    
    gzts = np.zeros(iters)
    for i in range(0, iters):
        gpu_u = torch.randn((1,1,M, N), dtype=torch.float, device='cuda')
        time_begin = time.time()
        _ = fd_gpu.ddz(gpu_u)
        torch.cuda.synchronize()  # CRITICAL: Wait for GPU to finish
        gzts[i] = time.time() - time_begin

    cxt = np.median(cxts)
    czt = np.median(czts)
    gxt = np.median(gxts)
    gzt = np.median(gzts)

    logger.info(f"CPU x time: {cxt * 1e3} ms")
    logger.info(f"CPU z time: {czt * 1e3} ms")
    logger.info(f"GPU x time: {gxt * 1e3} ms")
    logger.info(f"GPU z time: {gzt * 1e3} ms")

    logger.info(f"CPU/GPU time x {cxt / gxt}")
    logger.info(f"CPU/GPU time z {czt / gzt}")
