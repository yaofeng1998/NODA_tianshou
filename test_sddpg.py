import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pdb
from torchviz import make_dot
import matplotlib.pyplot as plt
import time

from sddpg import SDDPGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.exploration import GaussianNoise
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.continuous import Actor, Critic
from ODENet import ODEfunc
from ODENet import ODEBlock


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--simulator-lr', type=float, default=1e-4)
    parser.add_argument('--n-simulator-step', type=int, default=10)
    parser.add_argument('--loss-weight-trans', type=float, default=1)
    parser.add_argument('--loss-weight-rew', type=float, default=1)
    parser.add_argument('--simulator-loss-threshold', type=float, default=0.1)
    parser.add_argument('--simulator-hidden-dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--test-noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--ignore-done', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def visualize_network(args, model):
    state_shape = args.state_shape
    action_shape = args.action_shape
    if type(args.state_shape) is tuple:
        state_shape = args.state_shape[0]
    if type(args.action_shape) is tuple:
        action_shape = args.action_shape[0]
    vis_graph = make_dot(model(np.ones((1, state_shape)),
                               np.ones((1, action_shape))), params=dict(model.named_parameters()))
    vis_graph.view()


def test_sddpg(args=get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN
    env = gym.make(args.task)
    if args.task == 'Pendulum-v0':
        env.spec.reward_threshold = -250
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, device=args.device)
    actor = Actor(
        net, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(args.layer_num, args.state_shape,
              args.action_shape, concat=True, device=args.device)
    critic = Critic(net, args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    ode = ODEBlock(ODEfunc, args.state_shape, args.action_shape,
                   args.simulator_hidden_dim, args.device).to(args.device)
    ode_optim = torch.optim.Adam(ode.parameters(), lr=args.simulator_lr)
    policy = SDDPGPolicy(
        actor, actor_optim, critic, critic_optim, ode, ode_optim, args,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        ignore_done=args.ignore_done,
        estimation_step=args.n_step)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(
        policy, test_envs, action_noise=GaussianNoise(sigma=args.test_noise))
    # log
    log_path = os.path.join(args.logdir, args.task, 'sddpg')
    writer = SummaryWriter(log_path)

    def train_fn(x, global_step):
        simulator_loss_history = np.array(policy.simulator_loss_history)
        if len(simulator_loss_history) == 0:
            return None
        x = np.arange(len(simulator_loss_history))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].semilogy(x, simulator_loss_history[:, 0])
        ax[1].semilogy(x, simulator_loss_history[:, 1])
        fig.tight_layout()
        plt.savefig(log_path + str(time.time()) + ".pdf")
        return None

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer)
    assert stop_fn(result['best_reward'])
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    test_sddpg()
