import argparse
import subprocess

import matplotlib.pyplot as plt
import numpy

from data_loader import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--fontsize', type=int, default=24)
    parser.add_argument('--linewidth', type=int, default=3)
    parser.add_argument('--mode', default='test')
    args = parser.parse_args()

    loader = DataLoader()
    loader.set_data('saved.h5', args.epoch)
    truth = loader.get_value([args.mode, 'other_motion', 'input'])
    mirror = loader.get_value([args.mode, 'other_motion', 'prediction'])

    save_dir = '{:05d}/{:s}/motion/'.format(args.epoch, args.mode)
    plot_save_dir = save_dir + 'plot/'

    subprocess.check_output(['mkdir', '-p', save_dir])
    subprocess.check_output(['mkdir', '-p', plot_save_dir])

    corr_mat = numpy.corrcoef(
        numpy.concatenate(
            [truth.reshape(-1, 2), mirror.reshape(-1, 2)], axis=1).T)

    with open(save_dir + 'correlation.txt', 'w') as f:
        neurons = [
            'other neuron #1', 'other neuron #2', 'motion generator neuron #1',
            'motion generator neuron #2'
        ]
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                f.write('{:s} vs. {:s}: {:.3f} ({:f})'.format(
                    neurons[i], neurons[j], corr_mat[i, j], corr_mat[i, j]))
                f.write('\n')
        f.flush()

    def do_plot(idx):
        for i in range(2):
            ax = plt.subplot(2, 1, i + 1)
            ax.plot(truth[idx, :, i], c='r', lw=args.linewidth)
            ax.plot(mirror[idx, :, i], c='b', lw=args.linewidth)
            ax.set_xlim([0, truth.shape[1]])
            ax.set_ylim([-1.1, 1.1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', direction='in', labelsize=args.fontsize)
            ax.tick_params(axis='y', direction='in', labelsize=args.fontsize)
            ax.set_yticks([-1, 0, 1])
            ax.set_xlabel('Time step', fontsize=args.fontsize)
            ax.set_ylabel('Neuron {:d}'.format(i + 1), fontsize=args.fontsize)
        plt.tight_layout()

    for idx in range(truth.shape[0]):
        print(idx)
        do_plot(idx)
        # plt.show()
        plt.savefig(plot_save_dir + '{:05d}.eps'.format(idx))
        plt.savefig(plot_save_dir + '{:05d}.png'.format(idx))
        plt.clf()
