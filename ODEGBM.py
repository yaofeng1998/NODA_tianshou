import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
import lightgbm as lgb
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


class ODEGBM(nn.Module):

    def __init__(self, args, hidden_dim=20, tol=1e-3):
        super(ODEGBM, self).__init__()
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
        self.integration_time = torch.tensor([0, 1]).float()
        self.odefunc_obs = ODEfunc(hidden_dim)
        self.fc_state_out = nn.Linear(hidden_dim, state_shape)

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
        self.device = args.device
        self.tol = tol
        self.train_data = []
        self.train_targets = [[], []]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.simulator_lr)

    def get_obs(self, x):
        out_obs = self.fc_obs_in(x)
        self.integration_time = self.integration_time.type_as(out_obs)
        out_obs = odeint(self.odefunc_obs, out_obs, self.integration_time, rtol=self.tol, atol=self.tol)[1]
        # out_obs = self.odefunc_obs(0, out_obs)
        out_obs = self.fc_state_out(out_obs)

        return out_obs

    def train_sampled_obs(self):
        x_all = torch.cat(self.train_data)
        obs_target_all = torch.cat(self.train_targets[0])
        index = np.arange(len(x_all))
        for i in range(self.args.train_simulator_step):
            self.optimizer.zero_grad()
            np.random.shuffle(index)
            index = index[0: self.args.batch_size]
            x = x_all[index]
            out_obs = self.get_obs(x)
            obs_target = obs_target_all[index]
            loss_trans = self.args.loss_weight_trans * ((out_obs - obs_target) ** 2).mean()
            loss_trans.backward()
            self.optimizer.step()

    def train_GBM(self):
        lgb_train = lgb.Dataset(torch.cat(self.train_data).cpu().numpy(),
                                label=torch.cat(self.train_targets[1]).cpu().numpy())
        evals_result = {}
        self.gbm_model = lgb.train(self.gbm_parameters,
                                   lgb_train,
                                   valid_sets=[lgb_train],
                                   num_boost_round=2000,
                                   early_stopping_rounds=100,
                                   evals_result=evals_result,
                                   verbose_eval=50)
        return self.args.loss_weight_rew * np.mean(evals_result['training']['l2'])

    def forward(self, obs, act, train=True, targets=None, step=-1, **kwargs):
        concat_data = np.concatenate((obs, act), axis=1)
        x = torch.tensor(concat_data).float().to(self.device)
        out_obs = self.get_obs(x)
        if train:
            assert targets is not None
            loss_trans = self.args.loss_weight_trans * ((out_obs - targets[0]) ** 2).mean()
            loss_rew = 0
            self.train_data.append(x)
            self.train_targets[0].append(targets[0])
            self.train_targets[1].append(targets[1])
            self.train_sampled_obs()
            if step == 0:
                loss_rew = self.train_GBM()
            return loss_trans.item(), loss_rew
        else:
            assert self.gbm_model is not None
            return out_obs.cpu().numpy(), self.gbm_model.predict(concat_data,
                                                                          num_iteration=self.gbm_model.best_iteration)

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
