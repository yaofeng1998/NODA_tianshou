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
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.fc1(x)
        x = self.conv1(x)
        # x = self.bn(x)
        x = self.pooling(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class FCBlock(nn.Module):

    def __init__(self, args, original_dim=3, hidden_dim=2):
        super(FCBlock, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(original_dim, original_dim)
        self.fc2 = nn.Linear(original_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x


class NODAE(nn.Module):

    def __init__(self, args, hidden_dim=20, tol=1e-3):
        super(NODAE, self).__init__()
        self.args = args
        state_shape = args.state_shape
        action_shape = args.action_shape
        if type(state_shape) is tuple:
            state_shape = state_shape[0]
        if type(action_shape) is tuple:
            action_shape = action_shape[0]
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.block_num = 2
        self.fc_obs_in = nn.Linear(state_shape + action_shape, hidden_dim)
        self.encoder = FCBlock(args, original_dim=args.state_shape, hidden_dim=2)
        self.decoder = FCBlock(args, original_dim=2, hidden_dim=args.state_shape)
        self.integration_time = torch.tensor([0, 1]).float()
        self.odefunc_obs = ODEfunc(hidden_dim)
        self.fc_state_out = nn.Linear(hidden_dim, state_shape)

        hidden_dim = hidden_dim // 1
        self.fc_rew_in = nn.Linear(state_shape + action_shape, hidden_dim)
        self.odefunc_rew = ODEfunc(hidden_dim)
        self.rew_block = [ExpandBlock(hidden_dim)]
        for i in range(self.block_num):
            self.rew_block.append(BasicBlock(hidden_dim))
        self.rew_block = nn.Sequential(*self.rew_block)
        self.fc_rew_out = nn.Linear(hidden_dim, 1)
        self.device = args.device
        self.tol = tol
        self.train_data = []
        self.train_targets = [[], []]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.simulator_lr)

    def get_obs_rew(self, x):
        latent_s = self.encoder(x[:, 0:self.state_shape])
        self.integration_time = self.integration_time.to(args.device)
        out_obs = odeint(self.odefunc_obs, latent_s, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        out_obs = self.decoder(out_obs)

        out_rew = self.fc_rew_in(x)
        out_rew = odeint(self.odefunc_rew, out_rew, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        # out_rew = self.odefunc_rew(0, out_rew)
        out_rew = self.rew_block(out_rew)
        out_rew = torch.max(out_rew, dim=1)[0]
        out_rew = self.fc_rew_out(out_rew)[:, 0]
        return out_obs, out_rew

    def train_sampled_data(self):
        x_all = torch.cat(self.train_data)
        obs_target_all = torch.cat(self.train_targets[0])
        rew_target_all = torch.cat(self.train_targets[1])
        index = np.arange(len(x_all))
        for i in range(self.args.train_simulator_step):
            self.optimizer.zero_grad()
            np.random.shuffle(index)
            index = index[0: self.args.batch_size]
            x = x_all[index]
            out_obs, out_rew = self.get_obs_rew(x)
            obs_target = obs_target_all[index]
            rew_target = rew_target_all[index]
            loss_trans = self.args.loss_weight_trans * ((out_obs - obs_target) ** 2).mean()
            loss_rew = self.args.loss_weight_rew * ((out_rew - rew_target) ** 2).mean()
            decoded_s = self.decoder(self.encoder(x[:, 0:self.state_shape]))
            loss_ae = self.args.loss_weight_trans * ((x[:, 0:self.state_shape] - decoded_s) ** 2).mean()
            pdb.set_trace()
            print(loss_trans.item(), loss_ae.item())
            loss = loss_trans + loss_rew + loss_ae
            loss.backward()
            self.optimizer.step()
            # print(loss.item())

    def forward(self, obs, act, train=True, targets=None, **kwargs):
        x = torch.tensor(np.concatenate((obs, act), axis=1)).float().to(self.device)
        out_obs, out_rew = self.get_obs_rew(x)
        if train:
            assert targets is not None
            loss_trans = self.args.loss_weight_trans * ((out_obs - targets[0]) ** 2).mean()
            loss_rew = self.args.loss_weight_rew * ((out_rew - targets[1]) ** 2).mean()
            self.train_data.append(x)
            self.train_targets[0].append(targets[0])
            self.train_targets[1].append(targets[1])
            self.train_sampled_data()
            return loss_trans.item(), loss_rew.item()
        else:
            return out_obs.cpu().numpy(), out_rew.cpu().numpy()

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
