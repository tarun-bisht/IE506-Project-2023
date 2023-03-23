from src.model import LAE, VAE
from project.encoders import LinearEncoder, LinearEncoderB
from project.decoders import LinearDecoder


def LinearLAE(input_shape, nh, nz, energy, num_steps, step_size, metropolis_hastings):
    nx = input_shape[0]
    size = input_shape[1]
    encoder = LinearEncoder(size, nx, nh, nz)
    decoder = LinearDecoder(size, nx, nh, nz)
    model = LAE(input_shape=input_shape, encoder=encoder, decoder=decoder, 
                energy=energy, num_steps=num_steps, 
                step_size=step_size, metropolis_hastings=metropolis_hastings)
    return model

def LinearVAE(input_shape, nh, nz, energy):
    nx = input_shape[0]
    size = input_shape[1]
    encoder = LinearEncoderB(size, nx, nh, nz)
    decoder = LinearDecoder(size, nx, nh, nz)
    model = VAE(input_shape=input_shape, encoder=encoder, decoder=decoder, energy=energy)
    return model




