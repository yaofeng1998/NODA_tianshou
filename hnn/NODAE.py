import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from utils import choose_nonlinearity
import pdb


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

    def forward(self, t, x):
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

    def __init__(self, input_dim, hidden_dim, latent_dim, learn_rate, nonlinearity='tanh', tol=1e-3):
        super(NODAE, self).__init__()
        self.ae_obs = MLPAutoencoder(input_dim, hidden_dim, latent_dim, nonlinearity=nonlinearity)
        self.integration_time = torch.tensor([0, 1]).float()
        self.odefunc_obs = MLP(latent_dim, hidden_dim, latent_dim)

        # self.ae_rew = MLPAutoencoder(input_dim, hidden_dim, latent_dim, nonlinearity=nonlinearity)
        # self.odefunc_rew = MLP(latent_dim, hidden_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=1e-5)
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.to(x.device)
        x = self.ae_obs.encode(x)
        x = odeint(self.odefunc_obs, x, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        x = self.ae_obs.decode(x)
        return x

    def forward_train(self, x, targets, train=True, return_scalar=True):
        if train:
            self.integration_time = self.integration_time.to(x.device)
            x = self.ae_obs.encode(x)
            x = odeint(self.odefunc_obs, x, self.integration_time, rtol=self.tol, atol=self.tol)[1]
            x = self.ae_obs.decode(x)
            loss = ((x - targets) ** 2).mean() + 1e-3 * ((self.ae_obs(x) - x) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.integration_time = self.integration_time.to(x.device)
                x = self.ae_obs.encode(x)
                x = odeint(self.odefunc_obs, x, self.integration_time, rtol=self.tol, atol=self.tol)[1]
                x = self.ae_obs.decode(x)
        if return_scalar:
            return ((x - targets) ** 2).mean()
        else:
            return ((x - targets) ** 2).mean(dim=1)
