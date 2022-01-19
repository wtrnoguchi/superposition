import argparse
import os
import subprocess

import h5py
import numpy

import constants

parser = argparse.ArgumentParser()
parser.add_argument('--targets', nargs='+', type=str, default=[''])
parser.add_argument('--save_as', default='default')

args = parser.parse_args()

targets = []
for target in args.targets:
    f_target = constants.SAVE_ROOT_DIR + target + '/' + constants.SAVE_DATA_NAME
    with h5py.File(f_target, 'r') as f_h5:
        dct = {}
        for mode in f_h5.keys():
            dct[mode] = {}
            for modal in f_h5[mode].keys():
                dct[mode][modal] = f_h5[mode][modal][()]
        targets.append(dct)

save_dir = constants.SAVE_ROOT_DIR + args.save_as + '/'
subprocess.check_output(['mkdir', '-p', save_dir])

save_path = save_dir + constants.SAVE_DATA_NAME
assert not os.path.isfile(save_path)

h5_dataset = h5py.File(save_path, 'w')


for mode in targets[0].keys():
    for modal in list(targets[0].values())[0].keys():
        data = []
        for target in targets:
            data.append(target[mode][modal])
        data = numpy.concatenate(data)

        data_h5 = h5_dataset.create_dataset(
            f'{mode}/{modal}', data.shape, dtype='f')

        data_h5[:] = data
