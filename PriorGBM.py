import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import lightgbm as lgb
from tqdm import tqdm
import pdb


class PriorGBM(nn.Module):

    def __init__(self, args, tol=1e-10):
        super(PriorGBM, self).__init__()
        self.args = args
        self.device = args.device
        self.tol = tol
        self.trans_relative_noise = args.trans_relative_noise
        self.m = nn.Parameter((1 + 1 * self.trans_relative_noise * torch.randn(1)).to(self.device))
        self.l = nn.Parameter((1 + 1 * self.trans_relative_noise * torch.randn(1)).to(self.device))
        self.g = nn.Parameter((10 + 10 * self.trans_relative_noise * torch.randn(1)).to(self.device))
        self.dt = nn.Parameter((0.05 + 0.05 * self.trans_relative_noise * torch.randn(1)).to(self.device))
        self.gbm_model = None
        self.gbm_parameters = {
            'task': 'train',
            'application': 'regression',
            'boosting_type': 'gbdt',
            'learning_rate': 3e-3,
            'num_leaves': 80,
            'min_data_in_leaf': 10,
            'metric': 'l2',
            'max_bin': 2048,
            'verbose': 1,
            'nthread': 16,
        }
        self.train_data = []
        self.train_targets = [[], []]
        # if self.args.device == 'cuda':
        #     self.gbm_parameters['device'] = 'gpu'
        #     self.gbm_parameters['gpu_platform_id'] = 0
        #     self.gbm_parameters['gpu_device_id'] = 0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.simulator_lr)

    def get_obs(self, x):
        theta = torch.atan(torch.abs(x[:, 1]) / (torch.abs(x[:, 0]) + self.tol))
        theta = torch.where((x[:, 0] >= 0) & (x[:, 1] < 0), -theta, theta)
        theta = torch.where((x[:, 0] < 0) & (x[:, 1] >= 0), np.pi - theta, theta)
        theta = torch.where((x[:, 0] < 0) & (x[:, 1] < 0), theta - np.pi, theta)
        new_theta_dot = x[:, 2] + (3 * self.g / (2 * self.l) * torch.sin(theta)
                                   + 3 / (self.m * self.l ** 2) * x[:, 3]) * self.dt
        new_theta = theta + new_theta_dot * self.dt
        new_theta_dot = torch.clamp(new_theta_dot, min=-8, max=8)
        out_obs = torch.stack((torch.cos(new_theta), torch.sin(new_theta), new_theta_dot), dim=1)
        return out_obs, theta

    def train_sampled_trans(self, steps=10):
        x_all = torch.tensor(np.concatenate(self.train_data)).float().to(self.device)
        obs_target_all = torch.tensor(np.concatenate(self.train_targets[0])).float().to(self.device)
        index = np.arange(len(x_all))
        for i in range(steps):
            self.optimizer.zero_grad()
            np.random.shuffle(index)
            index = index[0: self.args.batch_size]
            x = x_all[index]
            out_obs, _ = self.get_obs(x)
            obs_target = obs_target_all[index]
            loss_trans = self.args.loss_weight_trans * ((out_obs - obs_target) ** 2).mean()
            loss_trans.backward()
            self.optimizer.step()
            print(loss_trans.item())

    def forward(self, obs, act, white_box=False, train=True, targets=None, step=-1):
        if train:
            assert targets is not None
        concat_data = np.concatenate((obs, act), axis=1)
        x = torch.tensor(concat_data).float().to(self.device)
        out_obs, theta = self.get_obs(x)
        if white_box:
            cost = theta ** 2 + .1 * x[:, 2] ** 2 + .001 * (x[:, 3] ** 2)
            out_rew = -cost
            if train:
                self.train_sampled_trans()
                loss_trans = self.args.loss_weight_trans * ((out_obs - targets[0]) ** 2).mean()
                loss_rew = self.args.loss_weight_rew * ((out_rew - targets[1]) ** 2).mean()
                return loss_trans.item(), loss_rew.item()
            return out_obs.detach().cpu().numpy(), out_rew.detach().cpu().numpy()
        else:
            if train:
                loss_trans = ((out_obs - targets[0]) ** 2).mean()
                self.train_data.append(concat_data)
                self.train_targets[0].append(targets[0].cpu().numpy())
                self.train_targets[1].append(targets[1].cpu().numpy())
                self.train_sampled_trans()
                if step == 0:
                    lgb_train = lgb.Dataset(np.concatenate(self.train_data),
                                            label=np.concatenate(self.train_targets[1]))
                    evals_result = {}
                    self.gbm_model = lgb.train(self.gbm_parameters,
                                               lgb_train,
                                               valid_sets=[lgb_train],
                                               num_boost_round=1000,
                                               early_stopping_rounds=100,
                                               evals_result=evals_result,
                                               verbose_eval=50)
                    loss_rew = self.args.loss_weight_rew * np.mean(evals_result['training']['l2'])
                else:
                    loss_rew = 0
                return loss_trans.item(), loss_rew
            return out_obs.detach().cpu().numpy(), self.gbm_model.predict(concat_data,
                                                                          num_iteration=self.gbm_model.best_iteration)
