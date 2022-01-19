import numpy
import torch

import util


class SequenceLoader(object):
    def __init__(self, data, batch_begin, n_steps):
        self.data = data
        self.batch_begin = batch_begin

        self.n_steps = n_steps

    def __iter__(self):
        self.t = 0
        return self

    def __next__(self):

        if self.t == self.n_steps:
            raise StopIteration()

        x = {}
        x['self_motion'] = self.data['self_motion'][:, self.t]
        x['self_position'] = self.data['self_position'][:, self.t]
        x['self_vision'] = self.data['self_vision'][:, self.t]

        x['other_motion'] = self.data['other_motion'][:, self.t]
        x['other_position'] = self.data['other_position'][:, self.t]

        y = {}
        y['self_motion'] = self.data['self_motion'][:, self.t + 1]
        y['self_vision'] = self.data['self_vision'][:, self.t + 1]
        y['self_position'] = self.data['self_position'][:, self.t + 1]

        y['other_motion'] = self.data['other_motion'][:, self.t + 1]
        y['other_position'] = self.data['other_position'][:, self.t + 1]

        current_t = self.t
        self.t += 1

        return x, y, current_t


class DataLoader(object):
    def __init__(self, data, batch_size, shuffle, device):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.n_data = data['self_position'].shape[0]
        self.n_steps = data['self_position'].shape[1] - 1

    def num_flatten(self):
        return self.n_data, self.n_steps + 1

    def num_sequence(self):
        return self.n_data, self.n_steps

    def load_flatten(self):
        if not (hasattr(self, '_flatten')):
            self._flatten = FlattenLoader(self.data, self.batch_size,
                                          self.n_data, self.n_steps + 1,
                                          self.shuffle, self.device)
        return self._flatten

    def load_sequence(self):
        if not (hasattr(self, '_seq')):
            self._seq = SeqLoader(self.data, self.batch_size, self.n_data,
                                  self.n_steps, self.shuffle, self.device)
        return self._seq


class SeqLoader(object):
    def __init__(self, data, batch_size, n_data, n_steps, shuffle, device):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.n_data = n_data
        self.n_steps = n_steps

    def get_n_data(self):
        return self.n_data
    
    def __len__(self):
        return self.n_data // self.batch_size

    def __iter__(self):
        self.order = numpy.random.permutation(
            self.n_data) if self.shuffle else numpy.arange(self.n_data)
        self.i = 0
        return self

    def __next__(self):

        if self.i >= self.n_data:
            raise StopIteration()

        batch_begin = self.i
        batch_end = batch_begin + self.batch_size
        batch_index = self.order[batch_begin:batch_end]
        batch_index = numpy.sort(batch_index)  # required for h5py fancy-index

        batch = {}

        batch['self_vision'] = util.scale_vision(
            util.transpose_vision(self.data['self_vision'][batch_index]))
        batch['self_vision'] = torch.from_numpy(batch['self_vision']).to(
            self.device)

        batch['self_motion'] = self.data['self_motion'][batch_index]

        batch['self_motion'] = torch.from_numpy(batch['self_motion']).to(
            self.device)

        batch['self_position'] = self.data['self_position'][batch_index]

        batch['self_position'] = torch.from_numpy(batch['self_position']).to(
            self.device)

        batch['other_motion'] = self.data['other_motion'][batch_index]

        batch['other_motion'] = torch.from_numpy(batch['other_motion']).to(
            self.device)

        batch['other_position'] = self.data['other_position'][batch_index]

        batch['other_position'] = torch.from_numpy(batch['other_position']).to(
            self.device)

        self._batch_index = batch_index

        self.i += self.batch_size

        return SequenceLoader(batch, self.i, self.n_steps), self._batch_index


class FlattenLoader(object):
    def __init__(self, data, batch_size, n_data, n_steps, shuffle, device):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self._n_data = n_data
        self._n_steps = n_steps
        self.n_data = self._n_data * self._n_steps

    def __len__(self):
        return self.n_data // self.batch_size

    def __iter__(self):
        self.order = numpy.random.permutation(
            self.n_data) if self.shuffle else numpy.arange(self.n_data)
        self.i = 0
        return self

    def __next__(self):

        if self.i >= self.n_data:
            raise StopIteration()

        batch_begin = self.i
        batch_end = batch_begin + self.batch_size
        batch_index = self.order[batch_begin:batch_end]
        batch_index = numpy.sort(batch_index)  # required for h5py fancy-index

        batch_index = numpy.unravel_index(batch_index,
                                          (self._n_data, self._n_steps))

        batch = {}

        batch['self_vision'] = util.scale_vision(
            util.transpose_vision(
                self.data['self_vision'][batch_index[0], batch_index[1]],
                dim=4))
        batch['self_vision'] = torch.from_numpy(batch['self_vision']).to(
            self.device)

        # batch['self_motion'] = self.data['self_motion'][batch_index[0],
        #                                                 batch_index[1]]

        # batch['self_motion'] = torch.from_numpy(batch['self_motion']).to(
        #     self.device)

        batch['self_position'] = self.data['self_position'][batch_index[0],
                                                            batch_index[1]]

        batch['self_position'] = torch.from_numpy(batch['self_position']).to(
            self.device)

        if 'other_vision' in self.data.keys():
            batch['other_vision'] = util.scale_vision(
                util.transpose_vision(
                    self.data['other_vision'][batch_index[0], batch_index[1]],
                    dim=4))
            batch['other_vision'] = torch.from_numpy(batch['other_vision']).to(
                self.device)

        # batch['other_motion'] = self.data['other_motion'][batch_index[0],
        #                                                   batch_index[1]]

        # batch['other_motion'] = torch.from_numpy(batch['other_motion']).to(
        #     self.device)

        batch['other_position'] = self.data['other_position'][batch_index[0],
                                                              batch_index[1]]

        batch['other_position'] = torch.from_numpy(batch['other_position']).to(
            self.device)

        self.i += self.batch_size

        return batch, batch_index
