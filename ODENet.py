import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import pdb


class ODEfunc(nn.Module):

    def __init__(self, dim=4, hidden_dim=20):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, state_dim, action_dim, hidden_dim=20, device='cpu', tol=1e-3):
        super(ODEBlock, self).__init__()
        if type(state_dim) is tuple:
            state_dim = state_dim[0]
        if type(action_dim) is tuple:
            action_dim = action_dim[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_before_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_before_2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc_before_3 = nn.Linear(hidden_dim, hidden_dim)
        self.odefunc = odefunc(hidden_dim)
        self.fc_after_1 = nn.Linear(hidden_dim, state_dim)
        self.fc_after_2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.integration_time = torch.tensor([0, 1]).float()
        self.device = device
        self.tol = tol

    def forward(self, obs, act):
        x = torch.tensor(np.concatenate((obs, act), axis=1)).float().to(self.device)
        out = self.fc_before_1(x)
        out = self.bn1(out)
        out = self.fc_before_2(out)
        out = self.relu1(out)
        out = self.fc_before_3(out)
        self.integration_time = self.integration_time.type_as(out)
        out_ode = odeint(self.odefunc, out, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        out = self.fc_after_1(out_ode)
        out = out.view(-1, self.state_dim)
        out_rew = self.fc_after_2(out_ode).view(-1, 1)
        return out, out_rew

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


if __name__ == "__main__":
    A = ODEBlock(ODEfunc, 64, 32).cuda()
    B = torch.zeros((1, 64)).cuda()
    pdb.set_trace()
