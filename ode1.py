
import torch
from torch import nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):

    def __init__(self, adj, latent_dim):
        super(ODEFunc, self).__init__()
        self.g = adj
        self.alpha_train = 0.9*torch.ones(adj.shape[1]).to('cuda')

    def forward(self, t, x):
        alph = nn.functional.sigmoid(self.alpha_train).unsqueeze(dim=1)
        ax = torch.spmm(self.g, x)
        ax = alph*torch.spmm(self.g, ax)
        f =  ax-x
        return f
    
class ODEblock(nn.Module):
    def __init__(self, odefunc, t = torch.tensor([0,1]), solver='euler'):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.solver = solver

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method=self.solver)[1]
        return z