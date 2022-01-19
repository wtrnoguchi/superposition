import os

import h5py

import cv2
import util
from constants import SAVE_DATA_NAME


class DataSaver(object):
    def __init__(
            self,
            save_dir,
            mode,
            save_targets=[],
            save_targets_index=[],
            save_vision_img=False,
            img_ext='jpg',
    ):
        self.save_dir = save_dir
        self.mode = mode
        self.reset()
        self.save_targets = save_targets
        self.save_targets_index = save_targets_index
        self.save_vision_img = save_vision_img
        self.img_ext = img_ext
        util.mkdir_p(self.save_dir)

    def set_num(self, n_data, n_steps):
        self.n_data = n_data
        self.n_steps = n_steps

    def reset(self):
        self.data = {}

    def _recursive_add(self, idx0, idx1, parent, parent_path):
        parent_path

        for name, child in parent.items():
            path = parent_path + '/{:s}'.format(name)

            if isinstance(child, dict):
                self._recursive_add(idx0, idx1, child, path)
            elif isinstance(child, tuple):
                self._recursive_add(idx0, idx1, child._asdict(), path)
            else:
                if any(target in path for target in self.save_targets):
                    self._append(path, idx0, idx1, child)

    def add(self, epoch, idx0, idx1, data):

        path = '{:05d}/{:s}'.format(epoch, self.mode)
        self._recursive_add(idx0, idx1, data, path)

    def _append(self, name, idx0, idx1, data):

        paths = os.path.normpath(name).split(os.path.sep)
        paths = [p for p in paths if len(p) > 0]
        temp = self.data
        for i, p in enumerate(paths):
            if i == (len(paths) - 1):
                if not (p in temp.keys()):
                    temp[p] = []
            else:
                if not (p in temp.keys()):
                    temp[p] = {}
            temp = temp[p]
        temp.append(((idx0, idx1), data))

    def do_save(self):
        path = ''
        self._recursive_save(self.data, path)
        self.reset()

    def _save_vision(self, data, path):

        save_root = self.save_dir + path + '/'
        util.mkdir_p(save_root)
        for point in data:
            (idx0, idx1), data = point
            data = data.data.cpu().numpy()
            data = util.de_scale_vision(util.de_transpose_vision(data, dim=4))
            _idx0 = [idx0] * data.shape[0] if isinstance(idx0, int) else idx0
            _idx1 = [idx1] * data.shape[0] if isinstance(idx1, int) else idx1
            for i, (__idx0, __idx1) in enumerate(zip(_idx0, _idx1)):

                if len(self.save_targets_index) > 0:
                    if not (__idx0 in self.save_targets_index):
                        continue

                cv2.imwrite(
                    save_root + '{:05d}_{:05d}.{:s}'.format(
                        __idx0, __idx1, self.img_ext), data[i] * 255)

    def _save_value(self, data, path):

        with h5py.File(self.save_dir + SAVE_DATA_NAME, 'a') as f_h5:

            for point in data:
                (idx0, idx1), data = point
                data = data.data.cpu().numpy()
                if 'vision' in path:
                    data = util.de_scale_vision(
                        util.de_transpose_vision(data, dim=4))

                shp = tuple([self.n_data, self.n_steps] + list(data.shape[1:]))
                if not (path in f_h5):
                    f_h5.create_dataset(path, shp)

                try:
                    f_h5[path][idx0, idx1] = data
                except Exception:
                    for i, (i0, i1) in enumerate(zip(idx0, idx1)):
                        f_h5[path][i0, i1] = data[i]

    def _recursive_save(self, node, parent_path):
        parent_path

        for name, child in node.items():
            path = parent_path + '/{:s}'.format(name)

            if isinstance(child, list):

                if 'vision' in path:
                    if self.save_vision_img:
                        self._save_vision(child, path)

                self._save_value(child, path)

            else:
                self._recursive_save(child, path)

    def save(self):
        if self.save_state:
            self.do_save_state()
