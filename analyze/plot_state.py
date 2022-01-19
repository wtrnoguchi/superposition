import argparse
import os
import subprocess

import numpy

from analyzer import PCAAnalyzer
from data_loader import DataLoader
from visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dim_x', type=int, default=0)
    parser.add_argument('--dim_y', type=int, default=1)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--state_type', default='hidden')
    parser.add_argument('--plot', type=str, default='self_position')
    parser.add_argument('--plot_layer', type=str, default='self')
    parser.add_argument('--analyze_layer', type=str, default='self')
    parser.add_argument('--analyze_mode', type=str, default='eval')
    parser.add_argument('--pca_components', type=int, default=2)
    parser.add_argument('--field_size', type=int, default=10)
    parser.add_argument('--animation', action='store_true', default=False)
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()

    loader = DataLoader()
    loader.set_data('saved.h5', args.epoch)
    position = loader.get_value([args.mode, args.plot, 'truth'])
    visualizer = Visualizer(args.field_size)

    label_prefix = 'PC'
    savedir_root = '{epoch:05d}/{mode}/state/map_by_{analyze_layer}_{analyze_mode}/'.format(
        **vars(args))
    savedir_plot = savedir_root + '{plot_layer}/{plot}/'.format(**vars(args))

    pcadir = '{epoch:05d}/pca/{analyze_layer}_{analyze_mode}/'.format(
        **vars(args))

    pca_name = pcadir + 'pca.pkl'

    subprocess.check_output(['mkdir', '-p', savedir_root])
    subprocess.check_output(['mkdir', '-p', savedir_plot])
    subprocess.check_output(['mkdir', '-p', pcadir])

    analyzer = PCAAnalyzer(n_components=args.pca_components)

    is_base = not os.path.exists(pca_name)
    if is_base:
        analyzer.fit(
            loader.get_flatten_hidden(
                mode=args.mode,
                layer=args.plot_layer,
                state_type=args.state_type))
        analyzer.save(pca_name)
        numpy.savetxt(
            pcadir + 'pca_contribution.txt',
            analyzer.contribution_ratio,
            fmt='%.3f')
    else:
        analyzer.load(pca_name)
    # print(analyzer.contribution_ratio)

    h = analyzer.transform_sequence(
        loader.get_hidden(args.mode, args.plot_layer, args.state_type))

    dim_x = args.dim_x
    dim_y = args.dim_y

    base_img_name = pcadir + \
        'base_{:s}_{:d}_{:d}'.format(args.plot_layer, dim_x, dim_y)
    if args.animation:
        tmp_dir = 'tmp/'
        savedir_plot_anim = savedir_plot + 'animation/'
        subprocess.check_output(['mkdir', '-p', tmp_dir])
        subprocess.check_output(['mkdir', '-p', savedir_plot_anim])

        visualizer.init_plot()

        visualizer.put_label(
            label_prefix + str(dim_x + 1),
            label_prefix + str(dim_y + 1),
        )

        for t in range(h.shape[1]):
            # print(t)
            ds = visualizer.plot(
                h[args.idx, t, [dim_x, dim_y]][None, None, :],
                position,
                'point',
                analyzer.get_lim([dim_x, dim_y]),
            )

            visualizer.set_transparent()
            visualizer.save_png(tmp_dir + 'test_{:05d}'.format(t))
            [d.remove() for d in ds]

            subprocess.check_output([
                'composite',
                tmp_dir + 'test_{:05d}.png'.format(t),
                base_img_name + '.png',
                tmp_dir + 'test_{:05d}.png'.format(t, t),
            ])

        save_name = savedir_plot_anim + \
            '{:d}_{:d}_{:05d}.gif'.format(dim_x, dim_y, args.idx)
        subprocess.check_output(
            ['convert', '-delay', '5', tmp_dir + 'test_*', save_name])
        subprocess.check_output(['rm', '-r', tmp_dir])
    else:
        visualizer.init_plot()

        visualizer.put_label(
            label_prefix + str(dim_x + 1),
            label_prefix + str(dim_y + 1),
        )
        ds = visualizer.plot(
            h[:, :, [dim_x, dim_y]],
            position,
            args.plot,
            analyzer.get_lim([dim_x, dim_y]),
        )
        save_name = savedir_plot + '{:d}_{:d}'.format(dim_x, dim_y)
        visualizer.save_png(save_name)
        visualizer.save_eps(save_name)

        if args.mode == 'eval':
            if not (os.path.isfile(base_img_name + '.png')):
                subprocess.check_output([
                    'cp',
                    save_name + '.png',
                    base_img_name + '.png',
                ])
        # visualizer.save_eps(savedir_plot + '{:d}_{:d}'.format(dim_x, dim_y))
