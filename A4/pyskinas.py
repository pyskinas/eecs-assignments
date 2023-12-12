import torch
from torch import nn

def hello():
    print("Hello from pyskinas.py!")

def pf(x: torch.Tensor):
    """
    permute, and flatten HxW dimension.
    """
    return torch.flatten(torch.permute(x, (0, 2, 3, 1)), start_dim=1, end_dim=2)

def get_head_stem_layer( in_channels, out_channels):
    """
    Since both networks in the head are initialised the same way, with mean 0, std 0.01 for weights and 
    bias = 0, I though I'd write this to avoid redundant code, we'll see how it goes...

    Note, layers are always:
    3x3 conv, stride 1, pad 1
    weights ~ N(0, 0.01)
    bias = 0
    """
    conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    conv.weight = nn.Parameter(torch.randn_like(conv.weight) * 0.01)
    conv.bias = nn.Parameter(torch.zeros_like(conv.bias))
    return conv

def get_intersection(x1a : torch.Tensor, x2a : torch.Tensor, x1b : torch.Tensor, x2b : torch.Tensor):
    """
    Gets the 1 dimensional intersections between vectors of points.
    x1a, x2a are the x1 and x2 vectors of (x1,x2) pairs in a.
    x1a and x2a need to be column vectors (shape: [N, 1]).
    x1b and x2b need to be row/indiscriminate vectors (shape: [N,]).
    """
    return torch.min(
        torch.clamp(torch.min(x2a-x1b, -x1a+x2b), 0),
        torch.min(x2a-x1a, x2b-x1b)
    )

def get_squares(x1 : torch.Tensor, x2 : torch.Tensor, y1 : torch.Tensor, y2 : torch.Tensor):
    """
    Given 2 points on a square (x1,y1), (x2,y2) it'll calculate the size of the square.
    
    It's a pretty useless function, I wrote the function boilerplate before I figured out that 
    the function itself is super simple.
    """
    return (x2-x1)*(y2-y1)

def get_xywh(T : torch.Tensor):
    """
    Input is shape (N,4) for N coordinates in XYXY (bottom-left, top-right) format, and we want to put it in
    XYWH format (centre coord, width, height).
    """
    x1, y1, x2, y2 = T.unbind(dim=1)
    x = (x1+x2)/2
    y = (y1+y2)/2
    w = torch.abs(x2-x1)
    h = torch.abs(y2-y1)
    return (x, y, w, h)