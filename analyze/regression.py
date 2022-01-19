import argparse
import os
import subprocess

import numpy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from data_loader import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dim_x', type=int, default=0)
    parser.add_argument('--dim_y', type=int, default=1)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--state_type', default='hidden')
    args = parser.parse_args()

    loader = DataLoader()
    loader.set_data('saved.h5', args.epoch)

    savedir_root = 'regression/{mode}/'.format(**vars(args))
    subprocess.check_output(['mkdir', '-p', savedir_root])

    for target_so in ["self", "other"]:
        target = target_so + "_position"
        for layer_so in ["self", "other"]:

            s = loader.get_flatten_hidden(mode=args.mode, layer=layer_so, state_type=args.state_type)

            _p = loader.get_value([args.mode, target, 'truth'])
            p = _p.reshape(-1, 2)

            model = Ridge(alpha=1.0)
            model.fit(s, p)

            pred = model.predict(s)

            mse = mean_squared_error(p, pred)
            f_name = "{:05d}_layer-{}_terget-{}.txt".format(args.epoch, layer_so, target_so)
            numpy.savetxt(savedir_root + f_name, [mse])