import torch
import numpy as np
from copy import deepcopy
from torch.distributions import Independent, Normal
from typing import Any, Dict, Tuple, Union, Optional

from tianshou.policy import DDPGPolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as
import pdb
from gym import spaces
from gym.utils import seeding
from os import path
from Environments import SimulationEnv
import argparse
import time


class SSACPolicy(DDPGPolicy):
    """Implementation of Simulator-based Soft Actor-Critic.
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient, default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration,
        defaults to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy,
        defaults to True.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            actor_optim: torch.optim.Optimizer,
            critic1: torch.nn.Module,
            critic1_optim: torch.optim.Optimizer,
            critic2: torch.nn.Module,
            critic2_optim: torch.optim.Optimizer,
            simulator: Optional[torch.nn.Module],
            args,
            action_range: Tuple[float, float],
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[
                float, Tuple[float, torch.Tensor, torch.optim.Optimizer]
            ] = 0.2,
            reward_normalization: bool = False,
            ignore_done: bool = False,
            estimation_step: int = 1,
            exploration_noise: Optional[BaseNoise] = None,
            deterministic_eval: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(None, None, None, None, action_range, tau, gamma,
                         exploration_noise, reward_normalization, ignore_done,
                         estimation_step, **kwargs)
        if simulator is not None:
            self.simulator = simulator
        self.args = args
        self.simulation_env = None
        self.loss_history = []
        self.gbm_model = None
        self.simulator_buffer = ReplayBuffer(size=self.args.buffer_size)
        self.update_step = 0

        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self.start_simulation = False

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self, mode: bool = True) -> "SSACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()
        ):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def forward(  # type: ignore
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        if not self.args.baseline:
            obs = self.simulator.encode(obs)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            x = logits[0]
        else:
            x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        y = self._action_scale * (1 - y.pow(2)) + self.__eps
        log_prob = dist.log_prob(x).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)
        if self._noise is not None and self.training and not self.updating:
            act += to_torch_as(self._noise(act.shape), act)
        act = act.clamp(self._range[0], self._range[1])
        return Batch(
            logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def _target_q(
            self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            batch.act = to_torch_as(batch.act, a_)
            if self.args.baseline:
                target_q = torch.min(
                    self.critic1_old(batch.obs_next, a_),
                    self.critic2_old(batch.obs_next, a_),
                ) - self._alpha * obs_next_result.log_prob
            else:
                target_q = torch.min(
                    self.critic1_old(self.simulator.encode(batch.obs_next), a_),
                    self.critic2_old(self.simulator.encode(batch.obs_next), a_),
                ) - self._alpha * obs_next_result.log_prob
        return target_q

    def learn_batch(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)

        # critic 1
        if self.args.baseline:
            current_q1 = self.critic1(batch.obs, batch.act).flatten()
        else:
            current_q1 = self.critic1(self.simulator.encode(batch.obs), batch.act).flatten()
        target_q = batch.returns.flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()
        # critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # critic 2
        if self.args.baseline:
            current_q2 = self.critic2(batch.obs, batch.act).flatten()
        else:
            current_q2 = self.critic2(self.simulator.encode(batch.obs), batch.act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()
        # critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        a = obs_result.act
        if self.args.baseline:
            current_q1a = self.critic1(batch.obs, a).flatten()
            current_q2a = self.critic2(batch.obs, a).flatten()
        else:
            current_q1a = self.critic1(self.simulator.encode(batch.obs), a).flatten()
            current_q2a = self.critic2(self.simulator.encode(batch.obs), a).flatten()
        actor_loss = (self._alpha * obs_result.log_prob.flatten()
                      - torch.min(current_q1a, current_q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "la": actor_loss.item(),
            "lc": (critic1_loss.item() + critic2_loss.item()) / 2.0,
        }
        if self._is_auto_alpha:
            result["lal"] = alpha_loss.item()
            result["a"] = self._alpha.item()  # type: ignore

        return result

    def learn_batch_NODA(self, batch: Batch) -> Dict[str, float]:
        obs_result = self(batch)
        obs = batch.obs
        act = obs_result.act.clone().detach().cpu().numpy()
        x = torch.tensor(np.concatenate((obs, act), axis=1)).float().to(self.simulator.device)
        out_obs, out_rew, recon_obs = self.simulator.get_obs_rew(x)
        current_q = out_rew.flatten().detach()
        actor_loss = (self._alpha * obs_result.log_prob.flatten() - current_q).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        result = {
            "la": actor_loss.item(),
            "lc": 0,
        }
        return result

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        batch.obs += self.args.noise_obs * np.random.randn(*np.shape(batch.obs))
        batch.rew += self.args.noise_rew * np.random.randn(*np.shape(batch.rew))
        self.simulator.critic1 = self.critic1_old
        self.simulator.critic2 = self.critic2_old
        if self.args.baseline:
            return self.learn_batch(batch)
        self.update_step += 1
        simulator_loss = self.learn_simulator(batch)
        result = self.learn_batch(batch)
        result["lt"] = simulator_loss[0]
        result["lc"] = simulator_loss[1]
        if self.update_step >= self.args.switch_step:
            if self.update_step % 100 == 0 or len(self.simulator_buffer) == 0:
                self.simulate_environment()
            simulation_batch, indice = self.simulator_buffer.sample(self.args.batch_size)
            # simulation_batch = self.process_fn(simulation_batch, self.simulator_buffer, indice)
            simulation_batch.returns = torch.from_numpy(simulation_batch.rew).float().to(self.args.device)
            result = self.learn_batch(simulation_batch)
            self.post_process_fn(simulation_batch, self.simulator_buffer, indice)
            result["lst"] = simulator_loss[0]
            result["lsc"] = simulator_loss[1]
            self.loss_history.append([simulator_loss[0], simulator_loss[1], result["la"], result["lc"], 0, 0])
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
        for j in range(obs.shape[1]):
            for i in range(self.args.n_simulator_step):
                self.simulator_buffer.add(obs[i, j], act[i, j], rew[i, j], done[i, j], obs_next[i, j])
        return None

    def learn_simulator(self, batch: Batch, fix_encoder=False):
        target_obs = torch.from_numpy(batch.obs_next).float().to(self.args.device)
        target_rew = batch.returns
        targets = [target_obs, target_rew]
        losses = self.simulator(batch.obs, batch.act, train=True, targets=targets, fix_encoder=fix_encoder)
        del target_obs
        return losses[0], losses[1]
