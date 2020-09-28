import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Tuple, Union, Optional
import gym

from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.data import Collector, Batch, ReplayBuffer, to_torch_as
import pdb


class SimulationEnv(gym.Env):
    def __init__(self, model, action_space, observation_space, args):
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_low = self.observation_space.low
        self.observation_high = self.observation_space.high
        self.model = model
        self.obs = None
        self.max_step = args.n_simulator_step
        self.current_step = 0
        self.batch_size = args.batch_size

    def reset(self):
        self.obs = np.random.rand(self.batch_size, *self.observation_low.shape) * \
                   (self.observation_high - self.observation_low) + self.observation_low
        self.current_step = 0
        return self.obs

    def step(self, action):
        with torch.no_grad():
            obs, reward = self.model(self.obs, action)
        if self.current_step == self.max_step:
            done = np.array([True] * self.batch_size)
        else:
            done = np.array([False] * self.batch_size)
            self.current_step += 1
        info = {}
        self.obs = obs.cpu().numpy()
        return self.obs, reward.cpu().numpy()[:, 0], done, info


class SDDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.
    We combine DDPG with a model-based simulator.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param torch.nn.Module simulator: the simulator network for the environment.
    :param torch.optim.Optimizer simulator_optim: the optimizer for simulator network.
    :param args: the arguments.
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
        simulator_optim: Optional[torch.optim.Optimizer],
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
        if simulator is not None and simulator_optim is not None:
            self.simulator = simulator
            self.simulator_optim = simulator_optim
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
        self.simulator_loss_history = []

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

    def learn_batch(self, batch: Batch, simulator_loss) -> Dict[str, float]:
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
            "lt": simulator_loss[0].item(),
            "lr": simulator_loss[1].item(),
            "la": actor_loss.item(),
            "lc": critic_loss.item(),
        }

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        simulator_loss = self.learn_simulator(batch)
        result = self.learn_batch(batch, simulator_loss)
        if simulator_loss[0].item() + simulator_loss[1].item() < self.simulator_loss_threshold:
            simulator_result = self.learn_batch(self.simulate_environment(), simulator_loss)
            result["la2"] = simulator_result["la"]
            result["lc2"] = simulator_result["lc"]
        return result

    def simulate_environment(self):
        self.simulation_env = SimulationEnv(self.simulator,
                                            self.base_env.action_space,
                                            self.base_env.observation_space, self.args)
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
        batch = Batch(obs=obs[:-1], act=act, rew=rew, done=done, info=info, obs_next=obs[1:])
        batch.obs = batch.obs.reshape(-1, batch.obs.shape[-1])
        batch.act = batch.act.reshape(-1, batch.act.shape[-1])
        batch.rew = batch.rew.reshape(-1)
        batch.done = batch.done.reshape(-1)
        batch.obs_next = batch.obs_next.reshape(-1, batch.obs_next.shape[-1])
        batch = Batch(list(batch))
        batch = self.process_fn(batch, batch, np.arange(len(batch)))
        return batch

    def learn_simulator(self, batch: Batch):
        self.simulator_optim.zero_grad()
        trans_obs, rew = self.simulator(batch.obs, batch.act)
        target_trans_obs, target_rew = torch.tensor(batch.obs_next).float(), batch.returns
        target_trans_obs = target_trans_obs.to(trans_obs.device)
        target_rew = target_rew.to(rew.device)
        simulator_loss_trans = self.args.loss_weight_trans * \
                               ((trans_obs - target_trans_obs) ** 2).mean()
        simulator_loss_rew = self.args.loss_weight_rew * \
                             ((rew - target_rew) ** 2).mean()
        simulator_loss = simulator_loss_trans + simulator_loss_rew
        simulator_loss.backward()
        self.simulator_optim.step()
        # return (torch.abs(trans_obs - target_trans_obs) / (torch.abs(target_trans_obs) + 1e-6)).mean(), \
        #        (torch.abs(rew - target_rew) / (torch.abs(target_rew) + 1e-6)).mean()
        self.simulator_loss_history.append([simulator_loss_trans.item(), simulator_loss_rew.item()])
        return simulator_loss_trans, simulator_loss_rew
