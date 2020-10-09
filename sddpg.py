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


class SimulationEnv(gym.Env):
    def __init__(self, trans_model, rew_model, action_space, observation_space, args):
        self.env = gym.make(args.task)
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_low = self.observation_space.low
        self.observation_high = self.observation_space.high
        self.trans_model = trans_model
        self.rew_model = rew_model
        self.obs = None
        self.max_step = args.n_simulator_step
        self.current_step = 0
        self.batch_size = max(args.batch_size // args.n_simulator_step, 1)

    def reset(self):
        # self.obs = np.random.rand(self.batch_size, *self.observation_low.shape) * \
        #            (self.observation_high - self.observation_low) + self.observation_low
        # self.current_step = 0
        # return self.obs
        return self.env.reset()[np.newaxis, :]

    def step(self, action):
        # with torch.no_grad():
        #     obs, reward = self.trans_model(self.obs, action)
        # reward = self.rew_model.predict(np.concatenate((self.obs, action), axis=1),
        #                                 num_iteration=self.rew_model.best_iteration)
        # assert self.current_step <= self.max_step
        # if self.current_step == self.max_step:
        #     done = np.array([True] * self.batch_size)
        # else:
        #     done = np.array([False] * self.batch_size)
        #     self.current_step += 1
        # info = {}
        # self.obs = obs.cpu().numpy()
        # return self.obs, reward, done, info
        return self.env.step(action)


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
        self.gbm_model = None
        self.update_step = self.args.max_update_step
        self.gbm_parameters = {
            'task': 'train',
            'application': 'regression',
            'boosting_type': 'gbdt',
            'learning_rate': 3e-3,
            'num_leaves': 80,
            'min_data_in_leaf': 10,
            'metric': 'l2',
            'max_bin': 255,
            'verbose': -1,
            'nthread': 8,
        }
        if self.args.device == 'cuda':
            # self.gbm_parameters['device'] = 'gpu'
            # self.gbm_parameters['gpu_platform_id'] = 0
            # self.gbm_parameters['gpu_device_id'] = 0
            pass


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

    def learn_batch(self, batch: Batch, simulator_loss, simulating = False) -> Dict[str, float]:
        if simulating is False:
            if self.update_step == 0:
                return {
                    "lt": simulator_loss[0],
                    "lr": simulator_loss[1],
                }
            else:
                self.update_step -= 1
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
            "lt": simulator_loss[0],
            "lr": simulator_loss[1],
            "la": actor_loss.item(),
            "lc": critic_loss.item(),
        }

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self.update_step > 0:
            simulator_loss = self.learn_simulator(batch)
            result = self.learn_batch(batch, simulator_loss)
        else:
            simulator_loss = np.array([0, 0])
            result = {}
        if simulator_loss[0] + simulator_loss[1] <= self.simulator_loss_threshold:
            simulator_result = self.learn_batch(self.simulate_environment(), simulator_loss, True)
            result["la2"] = simulator_result["la"]
            result["lc2"] = simulator_result["lc"]
        return result

    def simulate_environment(self):
        self.simulation_env = SimulationEnv(self.simulator, self.gbm_model,
                                            self.base_env.action_space,
                                            self.base_env.observation_space, self.args)
        obs, act, rew, done, info = [], [], [], [], []
        obs.append(self.simulation_env.reset())
        for i in range(self.args.n_simulator_step):
            with torch.no_grad():
                act.append(self(Batch(obs=obs[-1], info={})).act.cpu().numpy())
            result = self.simulation_env.step(act[-1])
            obs.append(result[0].reshape(1,-1))
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
        batch.rew = self.gbm_model.predict(np.concatenate((batch.obs, batch.act), axis=1),
                                           num_iteration=self.gbm_model.best_iteration)
        batch = self.process_fn(batch, batch, np.arange(len(batch)))
        return batch

    def learn_simulator(self, batch: Batch):
        self.simulator_optim.zero_grad()
        trans_obs, rew = self.simulator(batch.obs, batch.act)
        target_trans_obs, target_rew = torch.tensor(batch.obs_next).float(), torch.tensor(batch.rew).float()
        target_trans_obs = target_trans_obs.to(trans_obs.device)
        target_rew = target_rew.to(rew.device)
        lgb_train = lgb.Dataset(np.concatenate((batch.obs, batch.act), axis=1), label=target_rew.cpu().numpy())
        evals_result = {}
        self.gbm_model = lgb.train(self.gbm_parameters,
                                   lgb_train,
                                   valid_sets=[lgb_train],
                                   num_boost_round=100,
                                   early_stopping_rounds=10,
                                   evals_result=evals_result,
                                   verbose_eval=False,
                                   init_model=self.gbm_model,
                                   keep_training_booster=True)
        simulator_loss_trans = self.args.loss_weight_trans * \
                               ((trans_obs - target_trans_obs) ** 2).mean()
        simulator_loss = simulator_loss_trans
        simulator_loss.backward()
        self.simulator_optim.step()
        simulator_loss_trans = simulator_loss_trans.item()
        simulator_loss_rew = self.args.loss_weight_rew * np.mean(evals_result['training']['l2'])
        self.simulator_loss_history.append([simulator_loss_trans, simulator_loss_rew])
        # return (torch.abs(trans_obs - target_trans_obs) / (torch.abs(target_trans_obs) + 1e-6)).mean().item(), \
        #        (torch.abs(rew - target_rew) / (torch.abs(target_rew) + 1e-6)).mean().item()
        return simulator_loss_trans, simulator_loss_rew
