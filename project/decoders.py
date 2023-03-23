import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, size, nx, nh, nz):
        super(LinearDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nx*size**2),
            nn.Unflatten(1, (nx, size, size))
        )

    def forward(self, z):
        return self.net(z)