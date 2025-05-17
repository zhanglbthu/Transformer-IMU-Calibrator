import torch

def add_gaussian_noise(x:torch.Tensor, sigma=1):
    """
    Add gaussian noise to a Tensor.
    :param x: Tensor
    :param sigma: Noise level (std)
    :return: Noised Tensor.
    """
    device = x.device
    noise = torch.randn(size=x.shape).to(device) * sigma
    return x + noise