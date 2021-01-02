import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from SSAC import SSACPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic
from PriorGBM import PriorGBM
from ODENet import ODENet
from ODEGBM import ODEGBM
from NODA import NODA
from Plot_tensorboard import sort_file_by_time
import pdb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--n-step', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--collect-per-step', type=int, default=4)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--pre-collect-step', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-layer-size', type=int, default=256)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--train-simulator-step', type=int, default=3)
    parser.add_argument('--simulator-latent-dim', type=int, default=16)
    parser.add_argument('--simulator-hidden-dim', type=int, default=256)
    parser.add_argument('--simulator-lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='NODA')
    parser.add_argument('--simulator-batch-size', type=int, default=1024)
    parser.add_argument('--white-box', action='store_true', default=False)
    parser.add_argument('--loss-weight-trans', type=float, default=1)
    parser.add_argument('--loss-weight-ae', type=float, default=1)
    parser.add_argument('--loss-weight-rew', type=float, default=1)
    parser.add_argument('--noise-obs', type=float, default=0.0)
    parser.add_argument('--noise-rew', type=float, default=0.0)
    parser.add_argument('--n-simulator-step', type=int, default=200)
    parser.add_argument('--switch-step', type=int, default=1000)
    parser.add_argument('--imagine-step', type=int, default=10)
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    args = parser.parse_args()
    if args.baseline:
        args.train_simulator_step = 0
        args.max_update_step = 2 * args.epoch * args.step_per_epoch + 1
    return args


def test_sac(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    if args.baseline:
        state_shape = args.state_shape
    else:
        state_shape = args.simulator_latent_dim
    net = Net(args.layer_num, state_shape, device=args.device,
              hidden_layer_size=args.hidden_layer_size)
    actor = ActorProb(
        net, args.action_shape, args.max_action, args.device, unbounded=True,
        hidden_layer_size=args.hidden_layer_size, conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.layer_num, state_shape, args.action_shape,
                 concat=True, device=args.device,
                 hidden_layer_size=args.hidden_layer_size)
    critic1 = Critic(
        net_c1, args.device, hidden_layer_size=args.hidden_layer_size
    ).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(args.layer_num, state_shape, args.action_shape,
                 concat=True, device=args.device,
                 hidden_layer_size=args.hidden_layer_size)
    critic2 = Critic(
        net_c2, args.device, hidden_layer_size=args.hidden_layer_size
    ).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    if args.model == 'ODEGBM':
        model = ODEGBM(args).to(args.device)
    elif args.model == 'PriorGBM':
        model = PriorGBM(args).to(args.device)
    elif args.model == 'NODA':
        model = NODA(args).to(args.device)
    else:
        assert args.model == 'ODENet'
        model = ODENet(args).to(args.device)

    policy = SSACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim, model, args,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma, alpha=args.alpha,
        estimation_step=args.n_step)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(
            args.resume_path, map_location=args.device
        ))
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    if args.baseline:
        if not os.path.exists(log_path + '/baseline/'):
            os.makedirs(log_path + '/baseline/')
        writer = SummaryWriter(log_path + '/baseline')
    else:
        writer = SummaryWriter(log_path)

    def watch():
        # watch agent's performance
        print("Testing agent ...")
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=args.render)
        pprint.pprint(result)

    def save_fn(policy):
        # torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        pass

    def stop_fn(mean_rewards):
        return False

    if args.watch:
        watch()
        exit(0)

    # trainer
    if args.pre_collect_step > 0:
        train_collector.collect(n_step=args.pre_collect_step, random=True)
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, args.update_per_step,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer,
        log_interval=args.log_interval)
    pprint.pprint(result)
    # watch()


if __name__ == '__main__':
    test_sac()
