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

        hidden_dim = hidden_dim // 1
        self.fc_rew_in = nn.Linear(state_dim + action_dim, hidden_dim)
        self.odefunc_rew = odefunc(hidden_dim)
        self.rew_block = [ExpandBlock(hidden_dim)]
        for i in range(self.block_num):
            self.rew_block.append(BasicBlock(hidden_dim))
        self.rew_block = nn.Sequential(*self.rew_block)
        self.fc_rew_out = nn.Linear(hidden_dim, 1)
        self.device = device
        self.tol = tol

    def forward(self, obs, act):
        x = torch.tensor(np.concatenate((obs, act), axis=1)).float().to(self.device)

        out_obs = self.fc_obs_in(x)
        self.integration_time = self.integration_time.type_as(out_obs)
        out_obs = odeint(self.odefunc_obs, out_obs, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        # out_obs = self.odefunc_obs(0, out_obs)
        out_obs = self.fc_state_out(out_obs)

        out_rew = self.fc_rew_in(x)
        out_rew = odeint(self.odefunc_rew, out_rew, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        # out_rew = self.odefunc_rew(0, out_rew)
        # out_rew = self.rew_block(out_rew)
        # out_rew = torch.max(out_rew, dim=1)[0]
        out_rew = self.fc_rew_out(out_rew)
        return out_obs, out_rew

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
