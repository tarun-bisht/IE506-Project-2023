import math
import torch.nn as nn
import src.model.lae as lae
import src.model.vae as vae


def __create_linear_blocks(input_shape, nh, nz):
    nx = input_shape[0]
    size = input_shape[1]
    enc_layers =[nn.Flatten(),
                nn.Linear(nx*size**2, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nz)]
    
    dec_layers = [nn.Linear(nz, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.LayerNorm(nh),
                nn.ReLU(True),
                nn.Linear(nh, nx*size**2),
                nn.Unflatten(1, (nx, size, size))]
    return enc_layers, dec_layers

def __create_conv_blocks(input_shape, hidden_filters, latent_dim):
    inp_shape_w = input_shape[1]/2**(len(hidden_filters))
    inp_shape_h = input_shape[2]/2**(len(hidden_filters))
    if not(input_shape[1]%2**(len(hidden_filters)) == 0 and input_shape[2]%2**(len(hidden_filters)) ==0):
        raise ValueError("Input Size when divided by number of hidden layers is not integer which will reconstruct wrong output shape")
    
    inp_shape_w = math.ceil(input_shape[1]/2**(len(hidden_filters)))
    inp_shape_h = math.ceil(input_shape[2]/2**(len(hidden_filters)))

    enc_layers = []
    enc_layers.append(nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_filters[0], kernel_size=3, stride=2, padding=1))
    enc_layers.append(nn.ReLU())
    for i in range(len(hidden_filters)-1):
        enc_layers.append(nn.Conv2d(in_channels=hidden_filters[i], out_channels=hidden_filters[i+1], kernel_size=3, stride=2, padding=1))
        enc_layers.append(nn.ReLU())
    enc_layers.append(nn.Flatten())
    size = hidden_filters[-1]*inp_shape_w*inp_shape_h
    enc_layers.append(nn.Linear(in_features=size, out_features=16))
    enc_layers.append(nn.Linear(in_features=16, out_features=latent_dim))
    dec_layers = []
    dec_layers.append(nn.Linear(in_features=latent_dim, out_features=size))
    dec_layers.append(nn.Unflatten(1, (hidden_filters[-1], inp_shape_w, inp_shape_h)))
    dec_layers.append(nn.ConvTranspose2d(in_channels=hidden_filters[-1], out_channels=hidden_filters[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
    dec_layers.append(nn.ReLU())
    for i in range(len(hidden_filters)-1, 0, -1):
        dec_layers.append(nn.ConvTranspose2d(in_channels=hidden_filters[i], out_channels=hidden_filters[i-1], kernel_size=(3, 3), stride=2, padding=1, output_padding=1))
        dec_layers.append(nn.ReLU())
    dec_layers.append(nn.Conv2d(in_channels=hidden_filters[0], out_channels=input_shape[0], kernel_size=(3, 3), padding='same'))
    return enc_layers, dec_layers

## Layers
def LinearLAE(input_shape, nh, nz, energy, num_steps, step_size, metropolis_hastings, out_activation=None):
    enc_layers, dec_layers = __create_linear_blocks(input_shape, nh, nz)
    enc = lae.Encoder(enc_layers)
    dec = lae.Decoder(dec_layers)
    model = lae.LAE(input_shape=input_shape, 
                    encoder=enc, decoder=dec, 
                    energy=energy, num_steps=num_steps, 
                    step_size=step_size, 
                    metropolis_hastings=metropolis_hastings,
                    out_activation=out_activation)
    return model

def LinearVAE(input_shape, nh, nz, energy, out_activation=None):
    enc_layers, dec_layers = __create_linear_blocks(input_shape, nh, nz)
    enc = vae.Encoder(enc_layers)
    dec = vae.Decoder(dec_layers)
    model = vae.VAE(input_shape=input_shape, encoder=enc, 
                    decoder=dec, energy=energy, 
                    out_activation=out_activation)
    return model

def ConvLAE(input_shape, hidden_filters, latent_dim, energy, num_steps, step_size, metropolis_hastings, out_activation=None):
    enc_layers, dec_layers = __create_conv_blocks(input_shape, hidden_filters, latent_dim)
    enc = lae.Encoder(enc_layers)
    dec = lae.Decoder(dec_layers)
    model = lae.LAE(input_shape=input_shape, 
                    encoder=enc, decoder=dec, 
                    energy=energy, num_steps=num_steps, 
                    step_size=step_size, 
                    metropolis_hastings=metropolis_hastings,
                    out_activation=out_activation)
    return model

def ConvVAE(input_shape, hidden_filters, latent_dim, energy, out_activation=None):
    enc_layers, dec_layers = __create_conv_blocks(input_shape, hidden_filters, latent_dim) 
    enc = vae.Encoder(enc_layers)
    dec = vae.Decoder(dec_layers)
    model = vae.VAE(input_shape=input_shape, encoder=enc, 
                    decoder=dec, energy=energy, out_activation=out_activation)
    return model
    

if __name__=="__main__":

    import torch
    from project.energy import MSEEnergy

    test_input = torch.randn(size=(64, 3, 256, 256))
    conv = ConvVAE(input_shape=(3, 256, 256), latent_dim=32, hidden_filters=[32, 64], energy=MSEEnergy())
    print(conv)
    print(conv(test_input))

