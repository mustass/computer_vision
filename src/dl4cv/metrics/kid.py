from torchmetrics.image.kid import KernelInceptionDistance
from torch import nn

class KID(nn.Module):
    def __init__(self,subset_size) -> None:
        super().__init__()
        self.kid = KernelInceptionDistance(num_subsets=subset_size)
    
    def __call__(self,real,fake):
        self.kid.reset()
        self.kid.update(real, real=True)
        self.kid.update(fake, real=False)
        kid_mean, _ = self.kid.compute()
        return kid_mean