import argparse
import os
import subprocess
import sys
import time

import cv2
from data_loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from simulation import creator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--mode', default='test')
    parser.add_argument(
        '--view', default='overview', choices=['self', 'other', 'overview'])
    args = parser.parse_args()

    env = creator.create_environment()

    env.init()
    env.reset()

    loader = DataLoader()
    loader.set_data('saved.h5', args.epoch)

    env.set_camera(args.view)

    time.sleep(1)  # this is necessary

    env.off_display()

    save_dir_base = '{epoch:05d}/{mode}/record/{view:s}/'.format(**vars(args))

    for it, sp in loader.get_value([args.mode, 'self_position']).items():
        op = loader.get_value([args.mode, 'other_position', it])
        save_dir = save_dir_base + '{it:s}/'.format(it=it)
        subprocess.check_output(['mkdir', '-p', save_dir])

        sp = sp[args.idx]
        op = op[args.idx]

        n_steps = sp.shape[0]

        for t in range(n_steps):
            env.set_agent_pos(sp[t], op[t])
            vision = env.capture()
            cv2.imwrite(save_dir + '{:05d}_{:05d}.png'.format(args.idx, t),
                        vision * 255)
