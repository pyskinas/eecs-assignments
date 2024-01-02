import torch
from torch import nn

def Linear(dim_in, dim_out):
    """
        Identical to nn.Linear(), except that weights sampled from a uniform 
        distribution bounded by [-c, c], where c = sqrt(6/(dim_in + dim_out)).
    """
    # Initialise linear layer.
    layer = nn.Linear(dim_in, dim_out)

    # Initialise c = sqrt(6/(dim_in + dim_out)).
    c = torch.sqrt(6/(torch.Tensor([dim_in]) + torch.Tensor([dim_out])))

    # Change weights to be sampled from a uniform distribution in [-c, c].
    layer.weight = nn.Parameter((torch.rand((dim_out, dim_in))-0.5)*2*c)

    # May not neet to change biases.
    layer.bias = nn.Parameter(torch.zeros((dim_out)))
    
    return layer