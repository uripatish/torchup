import torch
import torch.nn as nn
import torch.nn.functional as functional

class Normalize(nn.Module):

    def __init__(self, dim: int, p: int):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, inputs):
        outputs = functional.normalize(inputs, dim = self.dim, p = self.p)
        
        return outputs


