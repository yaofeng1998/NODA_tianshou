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


class ExpandBlock(nn.Module):

    def __init__(self, hidden_dim, expand_channels=5):
        super(ExpandBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.expand_channels = expand_channels
        self.conv1 = nn.Conv1d(1, self.expand_channels, 5, padding=2)
        self.bn = nn.BatchNorm1d(self.expand_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, hidden_dim, channels=5):
        super(BasicBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.fc1(x)
        x = self.conv1(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class ODEBlock(nn.Module):

    def __init__(self, odefunc, state_dim, action_dim, hidden_dim=20, device='cpu', tol=1e-3):
        super(ODEBlock, self).__init__()
        if type(state_dim) is tuple:
            state_dim = state_dim[0]
        if type(action_dim) is tuple:
            action_dim = action_dim[0]
        self.block_num = 3
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_obs_in = nn.Linear(state_dim + action_dim, hidden_dim)
        self.integration_time = torch.tensor([0, 1]).float()
        self.odefunc_obs = odefunc(hidden_dim)
        self.fc_state_out = nn.Linear(hidden_dim, state_dim)
        self.device = device
        self.tol = tol
        self.m = nn.Parameter(1 + torch.rand(1).to(self.device))
        self.l = nn.Parameter(1 + torch.rand(1).to(self.device))
        self.g = nn.Parameter(10 + torch.Tensor(1).to(self.device))
        self.dt = nn.Parameter(torch.rand(1).to(self.device))

    def forward(self, obs, act):
        x = torch.tensor(np.concatenate((obs, act), axis=1)).float().to(self.device)
        theta = torch.atan(torch.abs(x[:, 1]) / (torch.abs(x[:, 0]) + 1e-6))
        theta = torch.where((x[:, 0] >= 0) & (x[:, 1] < 0), -theta, theta)
        theta = torch.where((x[:, 0] < 0) & (x[:, 1] >= 0), np.pi - theta, theta)
        theta = torch.where((x[:, 0] < 0) & (x[:, 1] < 0), theta - np.pi, theta)
        new_theta_dot = x[:, 2] + (3 * self.g / (2 * self.l) * torch.sin(theta)
                                   + 3 / (self.m * self.l ** 2) * x[:, 3]) * self.dt
        new_theta = theta + new_theta_dot * self.dt
        out_obs = torch.stack((torch.cos(new_theta), torch.sin(new_theta), new_theta_dot), dim=1)
        return out_obs, out_obs[:, 0].unsqueeze(-1)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
