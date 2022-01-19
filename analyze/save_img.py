import argparse
import subprocess

import cv2
from data_loader import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--t', type=int, default=0)
    parser.add_argument('--mode', default='eval')
    args = parser.parse_args()

    loader = DataLoader()
    loader.set_data('saved.h5', args.epoch)

    save_dir_base = '{epoch:05d}/{mode}/'.format(**vars(args))

    for so in ['self_vision', 'other_vision']:
        for ir in ['reconstruction', 'input']:
            imgs = loader.get_value(['{:s}'.format(args.mode), so, ir])

            save_dir = save_dir_base + '/' + so + '/' + ir + '/'
            subprocess.check_output(['mkdir', '-p', save_dir])
            # print(imgs.shape)

            img = imgs[args.idx, args.t]

            cv2.imwrite(
                save_dir + '{:05d}_{:05d}.png'.format(args.idx, args.t),
                img * 255)
