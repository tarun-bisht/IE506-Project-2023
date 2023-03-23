import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, size, nx, nh, nz):
        super(LinearEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx*size**2, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nz)
        )
    def forward(self, x):
        return self.net(x)


class LinearEncoderB(nn.Module):
        def __init__(self, size, nx, nh, nz):
            super(LinearEncoderB, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx*size**2, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
            )
            self.mu = nn.Linear(nh, nz)
            self.logvar = nn.Linear(nh, nz)

        def forward(self, x):
            x = self.net(x)
            return self.mu(x), self.logvar(x)


