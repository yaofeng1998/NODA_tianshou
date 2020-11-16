import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional
import lightgbm as lgb
import gym

from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Collector, Batch, ReplayBuffer, to_torch_as
import pdb
from gym import spaces
from gym.utils import seeding
from os import path
from Environments import SimulationEnv
import argparse


class SDDPGPolicy(BasePolicy):
    """Implementation of Simulator-based Deep Deterministic Policy Gradient.
    We combine DDPG with a model-based simulator.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param torch.nn.Module simulator: the simulator network for the environment.
    :param argparse.Namespace args: the arguments.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param BaseNoise exploration_noise: the exploration noise,
        add to the action, defaults to ``GaussianNoise(sigma=0.1)``.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: Optional[torch.nn.Module],
        actor_optim: Optional[torch.optim.Optimizer],
        critic: Optional[torch.nn.Module],
        critic_optim: Optional[torch.optim.Optimizer],
        simulator: Optional[torch.nn.Module],
        args,
        action_range: Tuple[float, float],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        ignore_done: bool = False,
        estimation_step: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if actor is not None and actor_optim is not None:
            self.actor: torch.nn.Module = actor
            self.actor_old = deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim: torch.optim.Optimizer = actor_optim
        if critic is not None and critic_optim is not None:
            self.critic: torch.nn.Module = critic
            self.critic_old = deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim: torch.optim.Optimizer = critic_optim
        if simulator is not None:
            self.simulator = simulator
        self.args = args
        self.simulation_env = None
        self.simulator_loss_threshold = self.args.simulator_loss_threshold
        self.base_env = gym.make(args.task)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self._tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2.0
        self._action_scale = (action_range[1] - action_range[0]) / 2.0
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self.loss_history = []
        self.gbm_model = None
        self.update_step = self.args.max_update_step
        self.simulator_buffer = ReplayBuffer(size=self.args.buffer_size)

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True) -> "DDPGPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critic.train(mode)
        self.simulator.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
            self.critic_old.parameters(), self.critic.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        with torch.no_grad():
            target_q = self.critic_old(
                batch.obs_next,
                self(batch, model='actor_old', input='obs_next').act)
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.0
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, h = model(obs, state=state, info=batch.info)
        actions += self._action_bias
        if self._noise and not self.updating:
            actions += to_torch_as(self._noise(actions.shape), actions)
        actions = actions.clamp(self._range[0], self._range[1])
        return Batch(act=actions, state=h)

    def learn_batch(self, batch: Batch) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)
        current_q = self.critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        action = self(batch).act
        actor_loss = -self.critic(batch.obs, action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            "la": actor_loss.item(),
            "lc": critic_loss.item(),
        }

    def get_loss_batch(self, batch: Batch) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)
        with torch.no_grad():
            current_q = self.critic(batch.obs, batch.act).flatten()
            target_q = batch.returns.flatten()
            td = current_q - target_q
            critic_loss = (td.pow(2) * weight).mean()
            action = self(batch).act
            actor_loss = -self.critic(batch.obs, action).mean()
        return {
            "la": actor_loss.item(),
            "lc": critic_loss.item(),
        }

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self.update_step > 0:
            self.update_step -= 1
            simulator_loss = self.learn_simulator(batch)
            result = self.learn_batch(batch)
            result["lt"] = simulator_loss[0]
            result["lr"] = simulator_loss[1]
            # result["m"] = self.simulator.m
            # result["l"] = self.simulator.l
            # result["g"] = self.simulator.g
            # result["dt"] = self.simulator.dt
            self.loss_history.append([simulator_loss[0], simulator_loss[1], result["la"], result["lc"], 0, 0])
        else:
            result = self.get_loss_batch(batch)
            if self.simulator_buffer._size < self.args.batch_size and hasattr(self.simulator, 'train_sampled_data_rew'):
                self.simulator.train_sampled_data_rew()
            if kwargs['i'] == 0 or self.simulator_buffer._size < self.args.batch_size:
                self.simulate_environment()
            simulation_batch, indice = self.simulator_buffer.sample(self.args.batch_size)
            simulation_batch = self.process_fn(simulation_batch, self.simulator_buffer, indice)
            simulator_result = self.learn_batch(simulation_batch)
            self.post_process_fn(simulation_batch, self.simulator_buffer, indice)
            result["la2"] = simulator_result["la"]
            result["lc2"] = simulator_result["lc"]
            self.loss_history.append([0, 0, result["la"], result["lc"], result["la2"], result["lc2"]])
        return result

    def simulate_environment(self):
        self.simulation_env = SimulationEnv(self.args, self.simulator)
        obs, act, rew, done, info = [], [], [], [], []
        obs.append(self.simulation_env.reset())
        for i in range(self.args.n_simulator_step):
            with torch.no_grad():
                act.append(self(Batch(obs=obs[-1], info={})).act.cpu().numpy())
            result = self.simulation_env.step(act[-1])
            obs.append(result[0])
            rew.append(result[1])
            done.append(result[2])
            info.append(result[3])
        obs_next = np.array(obs[1:])
        obs = np.array(obs[:-1])
        act = np.array(act)
        rew = np.array(rew)
        done = np.array(done)
        # obs = obs.reshape(-1, obs.shape[-1])
        # act = act.reshape(-1, act.shape[-1])
        # rew = np.array(rew).reshape(-1)
        # done = np.array(done).reshape(-1)
        # obs_next = obs_next.reshape(-1, obs_next.shape[-1])
        # rew = rew.reshape(obs.shape[0], obs.shape[1])
        for j in range(obs.shape[1]):
            for i in range(self.args.n_simulator_step):
                self.simulator_buffer.add(obs[i, j], act[i, j], rew[i, j], done[i, j], obs_next[i, j])
        return None

    def learn_simulator(self, batch: Batch):
        target_obs, target_rew = torch.tensor(batch.obs_next).float(), torch.tensor(batch.rew).float()
        target_obs = target_obs.to(self.args.device)
        target_rew = target_rew.to(self.args.device)
        targets = [target_obs, target_rew]
        losses = self.simulator(batch.obs, batch.act, white_box=self.args.white_box, train=True,
                                targets=targets, step=self.update_step)
        return losses[0], losses[1]
