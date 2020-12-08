import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from tqdm import tqdm
import pdb


def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


class MLP(torch.nn.Module):
    '''Just a salt-of-the-earth MLP'''

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)


class MLPAutoencoder(torch.nn.Module):
    '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''

    def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
        super(MLPAutoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

        for l in [self.linear1, self.linear2, self.linear3, self.linear4,
                  self.linear5, self.linear6, self.linear7, self.linear8]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def encode(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = h + self.nonlinearity(self.linear2(h))
        h = h + self.nonlinearity(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        h = self.nonlinearity(self.linear5(z))
        h = h + self.nonlinearity(self.linear6(h))
        h = h + self.nonlinearity(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class NODAE(nn.Module):

    def __init__(self, args, tol=1e-3):
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
        self.latent_dim = args.simulator_latent_dim
        self.hidden_dim = args.simulator_hidden_dim
        self.ae = MLPAutoencoder(state_shape, self.hidden_dim, self.latent_dim, nonlinearity='relu')
        self.integration_time = torch.tensor([0, 1]).float()
        self.odefunc = MLP(self.latent_dim + action_shape, self.hidden_dim, self.latent_dim)
        self.rew_nn = MLP(self.latent_dim + action_shape, self.hidden_dim, 1)
        self.device = args.device
        self.tol = tol
        self.train_data = []
        self.train_targets = [[], []]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.simulator_lr, weight_decay=1e-5)

    def get_obs_rew(self, x):
        latent_s = self.ae.encode(x[:, 0:self.state_shape])
        self.integration_time = self.integration_time.to(self.device)
        def odefunc(t, input):
            return self.odefunc(torch.cat((input, x[:, self.state_shape:]), dim=1))
        out_obs = odeint(odefunc, latent_s, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        out_obs = self.ae.decode(out_obs)
        recon_obs = self.ae.decode(latent_s)
        out_rew = self.rew_nn(torch.cat((latent_s, x[:, self.state_shape:]), dim=1)).squeeze(-1)
        return out_obs, out_rew, recon_obs

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
            obs = x[:, 0:self.state_shape]
            out_obs, out_rew, recon_obs = self.get_obs_rew(x)
            obs_target = obs_target_all[index]
            rew_target = rew_target_all[index]
            loss_trans = self.args.loss_weight_trans * ((out_obs - obs_target) ** 2).mean()
            loss_ae = self.args.loss_weight_ae * ((recon_obs - obs) ** 2).mean()
            loss_rew = self.args.loss_weight_rew * ((out_rew - rew_target) ** 2).mean()
            loss = loss_trans + loss_rew + loss_ae
            loss.backward()
            self.optimizer.step()
            # print(loss.item())

    def forward(self, obs, act, train=True, targets=None, **kwargs):
        if self.args.task == 'Pendulum-v0':
            act = np.clip(act, -2, 2)
        x = torch.tensor(np.concatenate((obs, act), axis=1)).float().to(self.device)
        out_obs, out_rew, recon_obs = self.get_obs_rew(x)
        if self.args.task == 'Pendulum-v0':
            # pass
            # out_obs_norm = out_obs[:, 0] ** 2 + out_obs[:, 1] ** 2 + np.finfo(np.float32).eps
            # out_obs[:, 0] /= out_obs_norm
            # out_obs[:, 1] /= out_obs_norm
            out_obs[:, [0, 1]] = torch.clamp(out_obs[:, [0, 1]], -1, 1)
            out_obs[:, 2] = torch.clamp(out_obs[:, 2], -8, 8)
            # recon_obs_norm = recon_obs[:, 0] ** 2 + recon_obs[:, 1] ** 2 + np.finfo(np.float32).eps
            # recon_obs[:, 0] /= recon_obs_norm
            # recon_obs[:, 1] /= recon_obs_norm
            recon_obs[:, [0, 1]] = torch.clamp(recon_obs[:, [0, 1]], -1, 1)
            recon_obs[:, 2] = torch.clamp(recon_obs[:, 2], -8, 8)
        if train:
            assert targets is not None
            tensor_obs = torch.tensor(obs).to(self.device)
            loss_trans = self.args.loss_weight_trans * ((out_obs - targets[0]) ** 2).mean()
            loss_ae = self.args.loss_weight_ae * ((recon_obs - tensor_obs) ** 2).mean()
            loss_rew = self.args.loss_weight_rew * ((out_rew - targets[1]) ** 2).mean()
            self.train_data.append(x)
            self.train_targets[0].append(targets[0])
            self.train_targets[1].append(targets[1])
            self.train_sampled_data()
            return (loss_trans + loss_ae).item(), loss_rew.item()
        else:
            return out_obs.cpu().numpy(), out_rew.cpu().numpy()
