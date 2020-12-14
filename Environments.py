from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional
import gym
import numpy as np
import torch

from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Collector, Batch, ReplayBuffer, to_torch_as
import pdb
from gym import spaces
from gym.utils import seeding


class PendulumEnv(gym.Env):

    def __init__(self,args, model, action_space, observation_space, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        self.seed()
        self.max_step = args.n_simulator_step
        self.current_step = 0
        self.batch_size = 8

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.current_step += 1
        assert self.current_step <= self.max_step
        th, thdot = self.state[:, 0], self.state[:, 1]  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.stack((newth, newthdot), axis=1)
        if self.current_step == self.max_step:
            done = np.array([True] * self.batch_size)
        else:
            done = np.array([False] * self.batch_size)
        return self._get_obs(), -costs, done, {}

    def reset(self):
        self.current_step = 0
        high = np.array([np.pi, 1])
        state = []
        for i in range(self.batch_size):
            state.append(self.np_random.uniform(low=-high, high=high))
        self.state = np.stack(tuple(state), axis=0)
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state[:, 0], self.state[:, 1]
        return np.stack((np.cos(theta), np.sin(theta), thetadot), axis=1)

    @staticmethod
    def angle_normalize(x):
        return (x + np.pi) % (2 * np.pi) - np.pi


class SimulationEnv(gym.Env):
    def __init__(self, args, model):
        self.args = args
        self.white_box = args.white_box
        self.model = model
        self.obs = None
        self.max_step = args.n_simulator_step
        self.current_step = 0
        self.batch_size = 1600 // args.n_simulator_step
        self.task = args.task
        self.original_env = gym.make(self.task)

    # def reset_pendulum_v0(self):
    #     high = np.array([np.pi, 1])
    #     temp = np.random.rand(self.batch_size, *high.shape) * 2 * high - high
    #     self.obs = np.stack((np.cos(temp[:, 0]), np.sin(temp[:, 0]), temp[:, 1]), axis=1)
    #     self.current_step = 0
    #     return self.obs

    def reset(self):
        obs = []
        for i in range(self.batch_size):
            obs.append(self.original_env.reset())
        self.obs = np.stack(obs, axis=0)
        return self.obs

    def step(self, action):
        with torch.no_grad():
            obs, rew = self.model(self.obs, action, white_box=self.white_box, train=False)
        self.current_step += 1
        assert self.current_step <= self.max_step
        if self.current_step == self.max_step:
            done = np.array([True] * self.batch_size)
        else:
            done = np.array([False] * self.batch_size)
        info = {}
        self.obs = obs
        return self.obs, rew, done, info