import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pdb


def sort_file_by_time(file_path):
    files = os.listdir(file_path)
    if not files:
        return
    else:
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return files


def main():
    log_dir = 'log/Pendulum-v0/sac/'
    files = sort_file_by_time(log_dir)[-2:]
    files = files[::-1]
    ea_sac = event_accumulator.EventAccumulator(log_dir + files[0])
    ea_sac.Reload()
    # print(ea_sac.scalars.Keys())
    rew_sac_item_mean = ea_sac.scalars.Items('train/rew')
    rew_sac_item_std = ea_sac.scalars.Items('train/rew_std')
    step_sac = []
    rew_sac_mean = []
    rew_sac_std = []
    assert len(rew_sac_item_mean) == len(rew_sac_item_std)
    for i in range(len(rew_sac_item_mean)):
        step_sac.append(rew_sac_item_mean[i].step)
        rew_sac_mean.append(rew_sac_item_mean[i].value)
        rew_sac_std.append(rew_sac_item_std[i].value)

    ea_ssac = event_accumulator.EventAccumulator(log_dir + files[1])
    ea_ssac.Reload()
    rew_ssac_item_mean = ea_ssac.scalars.Items('train/rew')
    rew_ssac_item_std = ea_ssac.scalars.Items('train/rew_std')
    step_ssac = []
    rew_ssac_mean = []
    rew_ssac_std = []
    assert len(rew_ssac_item_mean) == len(rew_ssac_item_std)
    for i in range(len(rew_ssac_item_mean)):
        step_ssac.append(rew_ssac_item_mean[i].step)
        rew_ssac_mean.append(rew_ssac_item_mean[i].value)
        rew_ssac_std.append(rew_ssac_item_std[i].value)

    step_sac = np.array(step_sac)
    rew_sac_mean = np.array(rew_sac_mean)
    rew_sac_std = np.array(rew_sac_std)
    step_ssac = np.array(step_ssac)
    rew_ssac_mean = np.array(rew_ssac_mean)
    rew_ssac_std = np.array(rew_ssac_std)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(step_sac, rew_sac_mean, label="SAC")
    ax.fill_between(step_sac, rew_sac_mean - rew_sac_std, rew_sac_mean + rew_sac_std, alpha=0.3)
    ax.plot(step_ssac, rew_ssac_mean, label="NODAE-SAC")
    ax.fill_between(step_ssac, rew_ssac_mean - rew_ssac_std, rew_ssac_mean + rew_ssac_std, alpha=0.3)
    pdb.set_trace()
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend(loc='best')
    plt.savefig('SAC-NODAE-comparison.pdf')
    plt.close()


if __name__ == '__main__':
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('font', size=14)
    main()
