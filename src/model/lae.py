import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

class LAE(nn.Module):
    # input_shape without batch_size
    def __init__(self, input_shape, encoder, decoder, energy, num_steps, step_size, metropolis_hastings=True, out_activation=None):
        super(LAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.energy = energy
        self.num_steps = num_steps
        self.step_size = step_size
        self.mh = metropolis_hastings
        self.out_activation = out_activation
        self.reset_parameters()
        self.encoder.net[-1].weight.requires_grad_(False)
        self.__z_shape = self.__find_z_shape(input_shape)
        self.latent_dim = torch.sum(torch.tensor(self.__z_shape)).item()
        self.latent_shape = self.__z_shape
        if out_activation == "sigmoid":
            self.out_activation = F.sigmoid
        elif out_activation == "tanh":
            self.out_activation = F.tanh
        else:
            self.out_activation = None
    
    def __find_z_shape(self, input_shape):
        test_sample = torch.randn(1, *input_shape, device=next(self.parameters()).device)
        z = self.encoder(test_sample)
        return z.shape[1:]

    def reset_parameters(self):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, inputs):
        if self.training:
            h = self.encoder.net[:-1](inputs)
            fi = self.encoder.net[-1].weight.data
            for _ in range(self.num_steps):
                fi = self.__langevin_step(inputs, h, fi, retain_graph=True)
            z = F.linear(h, fi)
            out = self.decoder(z)
            # update encoder weight
            self.encoder.net[-1].weight.data = fi.detach()
        else:
            mu = self.encoder(inputs)
            q_z = td.Independent(td.Normal(mu, 0.05), 1)
            z = q_z.sample()
            out = self.decoder(z)
        loss = self.energy.calculate(out, inputs)
        if self.out_activation is not None:
            out = self.out_activation(out)
        recon_loss = F.mse_loss(out, inputs)
        return {"output":out, "loss":loss, "reconstruction_loss": recon_loss}

    def __langevin_step(self, inputs, h, fi, retain_graph=False):
        fi.requires_grad_()
        z = F.linear(h, fi)
        out = self.decoder(z)
        loss = self.energy.calculate(out, inputs)
        grad = torch.autograd.grad(loss.sum(), fi, retain_graph=retain_graph)[0]
        mu = fi - self.step_size * grad
        sigma = torch.sqrt(torch.tensor(2 * self.step_size/inputs.size(0), device=mu.device))
        # q(fi_prime | fi)
        q_fpf = td.Independent(td.Normal(mu, sigma), 2)
        fi_p = q_fpf.sample().detach()
        if self.mh:
            fi_p.requires_grad_()
            z_p = F.linear(h, fi_p)
            out = self.decoder(z_p)
            loss_p = self.energy.calculate(out, inputs)
            grad = torch.autograd.grad(loss_p.sum(), fi_p)[0]
            mu = fi - self.step_size * grad
            sigma = torch.sqrt(torch.tensor(2 * self.step_size/inputs.size(0), device=mu.device))
            # q(fi | fi_prime)
            q_ffp = td.Independent(td.Normal(mu, sigma), 2)
            # calculate alpha
            alpha = torch.exp(loss.sum() - loss_p.sum() + q_ffp.log_prob(fi) - q_fpf.log_prob(fi_p))
            # calculate acceptance rate
            accept_rate = torch.min(torch.ones_like(alpha), alpha)
            # with probability delta update fi else fi will remain unchanged
            delta = torch.rand_like(accept_rate)
            if delta < accept_rate:
                fi = fi_p.detach()
            else:
                fi = fi.detach()
        else:
            fi = fi_p.detach()
        return fi

    def sample(self, num_points=1):
        z = torch.randn(num_points, *self.__z_shape, device=next(self.parameters()).device)
        out = self.decoder(z)
        if self.out_activation is not None:
            out = self.out_activation(out)
        return out
    
    def predict(self, inputs):
        mu = self.encoder(inputs)
        q_z = td.Independent(td.Normal(mu, 0.05), 1)
        z = q_z.sample()
        out = self.decoder(z)
        if self.out_activation is not None:
            out = self.out_activation(out)
        return out

    def encode(self, inputs):
        mu = self.encoder(inputs)
        return td.Independent(td.Normal(mu, 0.05), 1)
    
    def decode(self, inputs):
        return self.decoder(inputs)


class Encoder(nn.Module):

    def __init__(self, layers: list):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)


class Decoder(nn.Module):

    def __init__(self, layers: list):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.net(inputs)


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
                nn.Linear(nh, nz)
            )

        def forward(self, x):
            return self.net(x)


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
    model = LAE(input_shape=(1, 28, 28), encoder=encoder, decoder=decoder, energy=energy, num_steps=2, step_size=1e-4, metropolis_hastings=True)
    model.to(torch.device("cuda"))
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
    