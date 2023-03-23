import torch
import torch.nn as nn
import torch.distributions as td

class VAE(nn.Module):
    # input_shape without batch_size
    def __init__(self, input_shape, encoder, decoder, energy):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.energy = energy
        self.reset_parameters()
        self.__z_shape = self.__find_z_shape(input_shape)
    
    def __find_z_shape(self, input_shape):
        test_sample = torch.randn(1, *input_shape, device=next(self.parameters()).device)
        z = self.encoder(test_sample)
        return z[0].shape[1:]

    def reset_parameters(self):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, inputs):
        mu_z, log_sigma_z = self.encoder(inputs)
        sigma_z = torch.exp(0.5 * log_sigma_z)
        q_z = td.Independent(td.Normal(mu_z, sigma_z), 1)
        z = q_z.rsample()
        out = self.decoder(z)

        # calculate loss
        energy = self.energy.calculate(inputs, out)
        p_z = td.Independent(td.Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
        regularization = td.kl_divergence(q_z, p_z).mean()
        loss = energy + regularization

        return {"output":out, "loss":loss, "reconstruction_loss": energy, "elbo": loss}

    def sample(self, num_points=1):
        z = torch.randn(num_points, *self.__z_shape, device=next(self.parameters()).device)
        out = self.decoder(z)
        return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')

    from src.utils.energy import EnergyFunction
    class Encoder(nn.Module):
        def __init__(self, size, nx, nh, nz):
            super(Encoder, self).__init__()
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


    class Decoder(nn.Module):
        def __init__(self, size, nx, nh, nz):
            super(Decoder, self).__init__()
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
    
    class MSEEnergy(EnergyFunction):
        def __init__(self):
            self.loss = nn.MSELoss()

        def calculate(self, inputs, targets):
            return self.loss(inputs, targets)

    energy = MSEEnergy()
    encoder = Encoder(size=28, nx=1, nh=512, nz=8)
    decoder = Decoder(size=28, nx=1, nh=512, nz=8)
    model = VAE(input_shape=(1, 28, 28), encoder=encoder, decoder=decoder, energy=energy)
    
    test_sample = torch.randn(size=(8, 1, 28, 28), device=next(model.parameters()).device)
    # Test in eval mode
    model.eval()
    # forward method
    out = model(test_sample)
    print(out["output"].shape)
    print(out["loss"])
    # sample method
    sample = model.sample()
    print(sample.shape)
    # sample method multiple
    sample = model.sample(num_points=8)
    print(sample.shape)

    # Test in train mode
    model.train()
    # forward method
    out = model(test_sample)
    print(out["output"].shape)
    print(out["loss"])
    # sample method
    sample = model.sample()
    print(sample.shape)
    # sample method multiple
    sample = model.sample(num_points=8)
    print(sample.shape)
