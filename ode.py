
import torch
from torch import nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):

    def __init__(self, adj, latent_dim, data_name):
        super(ODEFunc, self).__init__()
        self.g =adj
        self.x0 = None
        self.name = data_name

        if self.name == "Cell_Phones_and_Accessories":
            self.linear2_u = nn.Linear(latent_dim, 1)

        else:
            self.linear2_u = nn.Linear(latent_dim, int(latent_dim/2))
            self.linear2_u_1 = nn.Linear(int(latent_dim/2), 1)

    def forward(self, t, x):

        if self.name == "Cell_Phones_and_Accessories":
            alph = nn.functional.sigmoid(self.linear2_u(x))
        else:
            alph = nn.functional.sigmoid(self.linear2_u_1(self.linear2_u(x)))
            
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