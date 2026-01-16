import SpatialScheme
import logging
import torch
import torch.nn.functional as ptf

logger = logging.getLogger(__name__)


class TimeScheme:
    def __init__(self, dt: float):
        self.dt = dt

    def step(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LeapFrog(TimeScheme):
    def __init__(self, dt: float, ):
        super().__init__(dt)

    def step(self, f1: torch.Tensor, f2: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        return f2 + self.dt * g2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    N = 6

    logger.info("CPU test")