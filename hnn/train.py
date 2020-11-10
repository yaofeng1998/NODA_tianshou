# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import torch, argparse

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLPAutoencoder
from hnn import HNN, PixelHNN
from data import get_dataset
from utils import L2_loss
from NODAE import NODAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchsummary
import pdb


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=28**2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=2, type=int, help='latent dimension of autoencoder')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--input_noise', default=0.0, type=float, help='std of noise added to HNN inputs')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=800, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='pixels', type=str, help='either "real" or "sim" data')
    # parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--retrain', action='store_true', default=False, help='whether not not retrain the models')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the data')
    parser.set_defaults(feature=True)
    return parser.parse_args()

'''The loss for this model is a bit complicated, so we'll
    define it in a separate function for clarity.'''


def pixelhnn_loss(x, x_next, model, return_scalar=True):
    # encode pixel space -> latent dimension
    z = model.encode(x)
    z_next = model.encode(x_next)

    # autoencoder loss
    x_hat = model.decode(z)
    ae_loss = ((x - x_hat)**2).mean(1)

    # hnn vector field loss
    noise = args.input_noise * torch.randn(*z.shape)
    z_hat_next = z + model.time_derivative(z + noise)  # replace with rk4
    hnn_loss = ((z_next - z_hat_next)**2).mean(1)

    # canonical coordinate loss
    # -> makes latent space look like (x, v) coordinates
    w, dw = z.split(1, 1)
    w_next, _ = z_next.split(1, 1)
    cc_loss = ((dw - (w_next - w))**2).mean(1)

    # sum losses and take a gradient step
    loss = ae_loss + cc_loss + 1e-1 * hnn_loss
    if return_scalar:
        return loss.mean()
    return loss


def train_hnn(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, nonlinearity='relu')
    model = PixelHNN(args.latent_dim, args.hidden_dim, autoencoder=autoencoder, nonlinearity=args.nonlinearity,
                     baseline=False)
    print("HNN has {} paramerters in total".format(sum(x.numel() for x in model.parameters() if x.requires_grad)))
    # if args.verbose:
    #     print("Training baseline model:" if args.baseline else "Training HNN model:")
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

    # get dataset
    data = get_dataset('pendulum', args.save_dir, verbose=True, seed=args.seed)

    x = torch.tensor(data['pixels'], dtype=torch.float32)
    test_x = torch.tensor(data['test_pixels'], dtype=torch.float32)
    next_x = torch.tensor(data['next_pixels'], dtype=torch.float32)
    test_next_x = torch.tensor(data['test_next_pixels'], dtype=torch.float32)

    # vanilla ae train loop
    stats = {'train_loss': [], 'test_loss': []}
    with tqdm(total=args.total_steps + 1) as t:
        for step in range(args.total_steps + 1):
            # train step
            ixs = torch.randperm(x.shape[0])[:args.batch_size]
            loss = pixelhnn_loss(x[ixs], next_x[ixs], model)
            loss.backward()
            optim.step()
            optim.zero_grad()

            train_loss = model.get_l2_loss(x, next_x).cpu().numpy()
            test_loss = model.get_l2_loss(test_x, test_next_x).cpu().numpy()
            stats['train_loss'].append([train_loss.mean(), train_loss.std()])
            stats['test_loss'].append([test_loss.mean(), test_loss.std()])
            t.set_postfix(train_loss='{:.9f}'.format(train_loss.mean()),
                          test_loss='{:.9f}'.format(test_loss.mean()))
            if args.verbose and step % args.print_every == 0:
                # run validation
                test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
                test_loss = pixelhnn_loss(test_x[test_ixs], test_next_x[test_ixs], model)
                print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
            t.update()

    train_dist = pixelhnn_loss(x, next_x, model, return_scalar=False)
    test_dist = pixelhnn_loss(test_x, test_next_x, model, return_scalar=False)
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'.
          format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                 test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))
    return model, stats


def train_NODAE(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available()

    # init model and optimizer
    model = NODAE(args.input_dim, args.hidden_dim, args.latent_dim, args.learn_rate,
                  nonlinearity=args.nonlinearity)
    if use_cuda:
        model = model.cuda()
    print("NODAE has {} paramerters in total".format(sum(x.numel() for x in model.parameters() if x.requires_grad)))

    # get dataset
    data = get_dataset('pendulum', args.save_dir, verbose=True, seed=args.seed)

    x = torch.tensor(data['pixels'], dtype=torch.float32)
    test_x = torch.tensor(data['test_pixels'], dtype=torch.float32)
    next_x = torch.tensor(data['next_pixels'], dtype=torch.float32)
    test_next_x = torch.tensor(data['test_next_pixels'], dtype=torch.float32)

    if use_cuda:
        x = x.cuda()
        test_x = test_x.cuda()
        next_x = next_x.cuda()
        test_next_x = test_next_x.cuda()
    # vanilla ae train loop
    stats = {'train_loss': [], 'test_loss': []}
    with tqdm(total=args.total_steps + 1) as t:
        for step in range(args.total_steps + 1):
            # train step
            ixs = torch.randperm(x.shape[0])[:args.batch_size]
            loss = model.forward_train(x[ixs], next_x[ixs])

            train_loss = model.forward_train(x, next_x, False, False).cpu().numpy()
            test_loss = model.forward_train(test_x, test_next_x, False, False).cpu().numpy()
            stats['train_loss'].append([train_loss.mean(), train_loss.std()])
            stats['test_loss'].append([test_loss.mean(), test_loss.std()])
            t.set_postfix(train_loss='{:.9f}'.format(train_loss.mean()),
                          test_loss='{:.9f}'.format(test_loss.mean()))
            if args.verbose and step % args.print_every == 0:
                # run validation
                test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
                test_loss = model.forward_train(test_x[test_ixs], test_next_x[test_ixs], False)
                print("step {}, train_loss {:.4e}, test_loss {:.4e}"
                      .format(step, loss.item(), test_loss.item()))
            t.update()

    train_dist = model.forward_train(x, next_x, False, False)
    test_dist = model.forward_train(test_x, test_next_x, False, False)
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))
    return model, stats


def plot_results(stats_hnn, stats_NODAE, total_length=None):
    hnn_train_mean, hnn_train_std, hnn_test_mean, hnn_test_std = tuple(stats_hnn)
    NODAE_train_mean, NODAE_train_std, NODAE_test_mean, NODAE_test_std = tuple(stats_NODAE)
    step = np.arange(1, len(hnn_train_mean) + 1)
    if total_length is not None:
        hnn_train_mean = hnn_train_mean[:total_length]
        hnn_train_std = hnn_train_std[:total_length]
        hnn_test_mean = hnn_test_mean[:total_length]
        hnn_test_std = hnn_test_std[:total_length]
        NODAE_train_mean = NODAE_train_mean[:total_length]
        NODAE_train_std = NODAE_train_std[:total_length]
        NODAE_test_mean = NODAE_test_mean[:total_length]
        NODAE_test_std = NODAE_test_std[:total_length]
        step = step[:total_length]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='row')
    axes = ax.flatten()
    axes[0].semilogy(step, hnn_train_mean, label="HNN (training)")
    axes[0].semilogy(step, NODAE_train_mean, label="NODAE (training)")
    axes[0].fill_between(step, hnn_train_mean - hnn_train_std, hnn_train_mean + hnn_train_std, alpha=0.3)
    axes[0].fill_between(step, NODAE_train_mean - NODAE_train_std, NODAE_train_mean + NODAE_train_std, alpha=0.3)
    axes[1].semilogy(step, hnn_test_mean, label="HNN (testing)")
    axes[1].semilogy(step, NODAE_test_mean, label="NODAE (training)")
    axes[1].fill_between(step, hnn_test_mean - hnn_test_std, hnn_test_mean + hnn_test_std, alpha=0.3)
    axes[1].fill_between(step, NODAE_test_mean - NODAE_test_std, NODAE_test_mean + NODAE_test_std, alpha=0.3)
    axes[0].set_xlabel('Step')
    axes[1].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    plt.savefig("HNN-NODAE-comparison.pdf")
    plt.close()


def train(args):
    _, stats_hnn = train_hnn(args)
    _, stats_NODAE = train_NODAE(args)
    hnn_train_mean = np.array(stats_hnn['train_loss'])[:, 0]
    hnn_train_std = np.array(stats_hnn['train_loss'])[:, 1]
    NODAE_train_mean = np.array(stats_NODAE['train_loss'])[:, 0]
    NODAE_train_std = np.array(stats_NODAE['train_loss'])[:, 1]
    hnn_test_mean = np.array(stats_hnn['test_loss'])[:, 0]
    hnn_test_std = np.array(stats_hnn['test_loss'])[:, 1]
    NODAE_test_mean = np.array(stats_NODAE['test_loss'])[:, 0]
    NODAE_test_std = np.array(stats_NODAE['test_loss'])[:, 1]
    stats_hnn = [hnn_train_mean, hnn_train_std, hnn_test_mean, hnn_test_std]
    stats_NODAE = [NODAE_train_mean, NODAE_train_std, NODAE_test_mean, NODAE_test_std]
    return stats_hnn, stats_NODAE


if __name__ == "__main__":
    args = get_args()
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('font', size=14)
    if args.retrain:
        stats_hnn, stats_NODAE = train(args)
        np.savez('results.npz', stats_hnn=stats_hnn, stats_NODAE=stats_NODAE)
    else:
        try:
            results = np.load('results.npz', allow_pickle=True)
            stats_hnn = results['stats_hnn']
            stats_NODAE = results['stats_NODAE']
        except:
            stats_hnn, stats_NODAE = train(args)
            np.savez('results.npz', stats_hnn=stats_hnn, stats_NODAE=stats_NODAE)
    plot_results(stats_hnn, stats_NODAE)
