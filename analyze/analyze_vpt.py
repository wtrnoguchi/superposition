import argparse
import math
import os
import subprocess
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy
from scipy import stats

import cv2
from data_loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from simulation import creator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--fontsize', type=int, default=18)
    parser.add_argument('--ytick_space', type=float, default=0.1)
    parser.add_argument('--mode', default='eval')
    parser.add_argument(
        '--with_coordinate', action='store_true', default=False)
    parser.add_argument('--self_str', default=' s ')
    parser.add_argument('--other_str', default=' o ')
    parser.add_argument('--margin', type=int, default=0)
    parser.add_argument('--save_vision', action='store_true', default=False)
    parser.add_argument('--plot_every', type=int, default=5)
    args = parser.parse_args()

    if args.save_vision:
        env = creator.create_environment()
        env.init()
        env.reset()
        env.set_camera('overview')
        time.sleep(1)
    else:
        env = None

    f_h5 = h5py.File('data.h5', 'r')['train']

    loader = DataLoader()
    loader.set_data('saved.h5', args.epoch)

    self_true = f_h5['self_vision_no_agent'][()]
    other_true = f_h5['other_vision_no_agent'][()]
    self_input = f_h5['self_vision'][()]
    other_input = f_h5['other_vision'][()]
    self_rec = loader.get_value(
        ['{:s}'.format(args.mode), 'self_vision', 'reconstruction'])
    other_rec = loader.get_value(
        ['{:s}'.format(args.mode), 'other_vision', 'reconstruction'])

    self_pos = loader.get_value(
        ['{:s}'.format(args.mode), 'self_position', 'input'])
    other_pos = loader.get_value(
        ['{:s}'.format(args.mode), 'other_position', 'input'])

    def mean_whc(v):
        return numpy.mean(numpy.mean(numpy.mean(v, -1), -1), -1)

    NN, NN, h, w, c = self_true.shape
    size = math.sqrt(NN)
    assert (size % 1.0 == 0.0)
    size = int(size)

    self_true = self_true.reshape(size, size, size, size, h, w, c)
    other_true = other_true.reshape(size, size, size, size, h, w, c)
    self_input = self_input.reshape(size, size, size, size, h, w, c)
    other_input = other_input.reshape(size, size, size, size, h, w, c)
    true_images = self_true[0, 0]
    self_rec = self_rec.reshape(size, size, size, size, h, w, c)
    other_rec = other_rec.reshape(size, size, size, size, h, w, c)
    self_pos = self_pos.reshape(size, size, size, size, -1)
    other_pos = other_pos.reshape(size, size, size, size, -1)

    save_dir = 'vpt/{:d}/'.format(args.margin)

    if args.margin > 0:
        m = args.margin
        true_images = true_images[m:-m, m:-m]
        self_true = self_true[m:-m, m:-m, m:-m, m:-m]
        other_true = other_true[m:-m, m:-m, m:-m, m:-m]
        self_input = self_input[m:-m, m:-m, m:-m, m:-m]
        other_input = other_input[m:-m, m:-m, m:-m, m:-m]
        self_rec = self_rec[m:-m, m:-m, m:-m, m:-m]
        other_rec = other_rec[m:-m, m:-m, m:-m, m:-m]
        self_pos = self_pos[m:-m, m:-m, m:-m, m:-m]
        other_pos = other_pos[m:-m, m:-m, m:-m, m:-m]
        size = size - 2 * m

    xmin, ymin = self_pos[0, 0, 0, 0]
    xmax, ymax = self_pos[0, 0, -1, -1]
    # print(xmin, ymin, xmax, ymax)

    if args.with_coordinate:
        save_dir += 'with_coordinate/'

    save_dir_hist = save_dir + 'histogram/'
    save_dir_map = save_dir + 'map/'

    subprocess.check_output(['mkdir', '-p', save_dir_hist])
    subprocess.check_output(['mkdir', '-p', save_dir_map])

    # histogram
    diffs = (
        ('self_true_and_self_rec', 'Self true \n vs. self rec.',
         numpy.fabs(self_true - self_rec)),
        ('other_true_and_other_rec', 'Other true \n vs. other rec.',
         numpy.fabs(other_true - other_rec)),
        ('self_true_and_other_rec', 'Self true \n vs. other rec.',
         numpy.fabs(self_true - other_rec)),
        ('other_true_and_self_rec', 'Other true \n vs. self rec.',
         numpy.fabs(other_true - self_rec)),
    )

    f = open(save_dir_hist + 'result.txt', 'w')

    labels = []
    means = []
    stds = []
    errors = []
    for k, label, v in diffs:
        # print(k)
        mn = mean_whc(v)

        mn = mn[numpy.logical_not(numpy.isnan(mn))]

        mn = mn.reshape(-1)
        errors.append(mn)
        labels.append(label)
        mean = numpy.mean(mn)
        std = numpy.std(mn)
        means.append(mean)
        stds.append(std)

        f.write('{:s}: mean: {:f}, std: {:f}'.format(k, mean, std))
        f.write('\n')

    f.write('\n')

    for i, (ki, _, _) in enumerate(diffs):
        for j, (kj, _, _) in enumerate(diffs):
            # print(errors[i].shape)
            # print(errors[j].shape)
            t, p = stats.ttest_ind(errors[i], errors[j], equal_var=False)
            # print(t, p)

            f.write('{:s} - {:s}: t: {:f}, p: {:f}'.format(ki, kj, t, p))
            f.write('\n')

    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(direction='in')
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(
        numpy.arange(len(labels)),
        means,
        yerr=stds,
        align='center',
        color='0.5',
        edgecolor='none',
        capsize=10)
    _ymin, _ymax = plt.ylim()
    plt.xticks(numpy.arange(len(labels)), labels, fontsize=14)
    plt.yticks(
        numpy.arange(_ymin, _ymax, args.ytick_space), fontsize=args.fontsize)
    plt.ylabel('Error', fontsize=args.fontsize)
    plt.ylim([_ymin, _ymax])
    plt.savefig(save_dir_hist + 'plot.png')
    plt.savefig(save_dir_hist + 'plot.eps')
    plt.savefig(save_dir_hist + 'plot.pdf')

    # done

    # error map

    def do_plot(true_map, rec, sx, sy, ox, oy):
        # print(ox, oy, sx, sy)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_rasterization_zorder(2)

        err = numpy.fabs(true_map - rec)
        err = mean_whc(err)

        # rotate and reverse y axis for correspondense to simulation
        err = numpy.rot90(err)
        err = err[::-1]
        sy = -sy
        oy = -oy

        plt.imshow(
            err,
            cmap='jet',
            extent=[xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5],
            zorder=0)
        cbar = plt.colorbar(ticks=[numpy.min(err), numpy.max(err)])
        cbar.outline.set_visible(False)
        cbar.ax.set_yticklabels(['Low', 'High'], fontsize=args.fontsize)
        cbar.set_label(
            'Error', fontsize=args.fontsize, labelpad=-args.fontsize)

        c = plt.Circle((sx, sy),
                       color=(1, 1, 1, 0.5),
                       edgecolor=(1, 1, 1, 0.5),
                       radius=1)
        ax.add_patch(c)
        c = plt.Circle((ox, oy),
                       color=(1, 1, 1, 0.5),
                       edgecolor=(1, 1, 1, 0.5),
                       radius=1)
        ax.add_patch(c)

        plt.text(
            sx,
            sy,
            args.self_str,
            ha='center',
            va='center',
            size=args.fontsize,
            color='k',
            zorder=1)

        plt.text(
            ox,
            oy,
            args.other_str,
            ha='center',
            va='center',
            size=args.fontsize,
            color='k',
            zorder=1)

        plt.xlim([-10, 10])
        plt.ylim([-10, 10])

        if args.with_coordinate:
            plt.text(
                -10,
                10,
                '(-10, 10)',
                ha='center',
                va='bottom',
                size=args.fontsize,
                color='k')
            plt.text(
                10,
                10,
                '(10, 10)',
                ha='center',
                va='bottom',
                size=args.fontsize,
                color='k')
            plt.text(
                10,
                -10,
                '(10, -10)',
                ha='center',
                va='top',
                size=args.fontsize,
                color='k')
            plt.text(
                -10,
                -10,
                '(-10, -10)',
                ha='center',
                va='top',
                size=args.fontsize,
                color='k')

        plt.tick_params(
            axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(
            axis='y', which='both', left=False, right=False, labelleft=False)

    def plot(sxi, syi, oxi, oyi):
        sx, sy = self_pos[0, 0, sxi, syi]
        ox, oy = other_pos[oxi, oyi, 0, 0]

        orec = other_rec[oxi, oyi, sxi, syi]
        srec = self_rec[oxi, oyi, sxi, syi]
        strue = true_images[sxi, syi]
        otrue = true_images[oxi, oyi]
        oinput = other_input[oxi, oyi, sxi, syi]
        sinput = self_input[oxi, oyi, sxi, syi]

        do_plot(
            true_images,
            orec,
            sx,
            sy,
            ox,
            oy,
        )

        # plt.show()
        save_base = save_dir_map + \
            'other_ae_self_{:d}_{:d}_other_{:d}_{:d}'.format(
                sxi, syi, oxi, oyi)
        plt.savefig(save_base + '_error_map.png')
        # plt.savefig(save_base + '.eps')
        plt.savefig(save_base + '_error_map.eps', rasterized=True)
        plt.savefig(save_base + '_error_map.pdf', rasterized=True, dpi=300)
        plt.close()

        if args.save_vision:
            env.set_agent_pos(numpy.array([sx, sy]), numpy.array([ox, oy]))
            overview = env.capture()

            cv2.imwrite(save_base + '_self_rec.png', srec * 255)
            cv2.imwrite(save_base + '_self_true.png', strue * 255)
            cv2.imwrite(save_base + '_self_input.png', sinput * 255)
            cv2.imwrite(save_base + '_other_rec.png', orec * 255)
            cv2.imwrite(save_base + '_other_true.png', otrue * 255)
            cv2.imwrite(save_base + '_other_input.png', oinput * 255)
            cv2.imwrite(save_base + '_overview.png', overview * 255)

    # plot(3, 15, 15, 3)
    for oxi in range(0, size, args.plot_every):
        for oyi in range(0, size, args.plot_every):
            for sxi in range(0, size, args.plot_every):
                for syi in range(0, size, args.plot_every):
                    plot(sxi, syi, oxi, oyi)

    # done
