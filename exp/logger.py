import os
import subprocess

import numpy

import util


class Logger(object):
    def __init__(self, save_dir, root):
        self.log_dir = save_dir + root + '/'
        self.logs = {}
        self.write_list = []
        util.mkdir_p(self.log_dir)
        self.sum_func = numpy.mean

    def _get_target(self, name):
        paths = os.path.normpath(name).split(os.path.sep)
        temp = self.logs
        _path = ''
        for i, p in enumerate(paths):
            if i == (len(paths) - 1):
                if not (p in temp.keys()):
                    temp[p] = {
                        'value': [],
                        'file':
                        open(self.log_dir + _path + '/{:s}.log'.format(p), 'w')
                    }
            else:
                _path += p + '/'
                if not (p in temp.keys()):
                    subprocess.check_output(
                        ['mkdir', '-p', self.log_dir + _path])
                    temp[p] = {}
            temp = temp[p]
        return temp

    def get_value(self, name):
        return self.sum_func(self._get_target(name)['value'])

    def _add(self, name, value):
        target = self._get_target(name)
        target['value'].append(value)

    def write_all(self, epoch):
        for write_target in self.logs.values():
            self._recursive_write(epoch, write_target)

    def _recursive_write(self, epoch, node):
        if 'file' in list(node.keys()):
            if not (node['file'] is None):
                value = self.sum_func(node['value'])
                node['file'].write('{:d}, {:f}\n'.format(epoch, value))
                node['file'].flush()
            node['value'] = []
        else:
            for child in node.values():
                self._recursive_write(epoch, child)

    def add(self, loss):
        self._recursive_add_log(loss)

    def _recursive_add_log(self, loss, root=None):

        root = root + '/' if root else ''
        for k in loss.keys():
            child = root + k
            if isinstance(loss[k], dict):
                self._recursive_add_log(loss[k], child)
            else:
                if isinstance(loss[k], int) or isinstance(loss[k], float):
                    _loss_value = loss[k]
                else:
                    _loss_value = loss[k].data.cpu().numpy().item()
                self._add(child, _loss_value)
