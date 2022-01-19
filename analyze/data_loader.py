import h5py

import analyze_util


class DataLoader(object):
    def _recursive_load(self, loaded, parent_name, parent):

        if isinstance(parent, h5py._hl.group.Group):
            if parent_name is None:
                _loaded = loaded
            else:
                loaded[parent_name] = {}
                _loaded = loaded[parent_name]
            for child_name, child in parent.items():
                self._recursive_load(_loaded, child_name, child)
        else:
            loaded[parent_name] = parent[()]

    def set_data(self, f_name, epoch):
        f_h5 = h5py.File(f_name, 'r')

        base = f_h5['/{epoch:05d}'.format(epoch=epoch)]
        self.data = {}
        self._recursive_load(self.data, None, base)

    def get_value(self, keys):
        data = self.data
        for k in keys:
            data = data[k]
        return data

    def get_self_position(self, mode):
        return self.data[mode]['self_position']['truth']

    def get_other_position(self, mode):
        return self.data[mode]['other_position']['truth']

    def get_hidden(self, mode, layer, state_type):
        return self.data[mode]['state'][layer][state_type]

    def get_flatten_hidden(self, mode=None, layer=None, state_type=None):
        hiddens = []
        for _m in self.data.keys() if mode is None else [mode]:
            state = self.data[_m]['state']
            l_keys = state.keys()
            for _l in l_keys if layer is None else [layer]:
                s_keys = state[_l].keys()
                for _s in s_keys if state_type is None else [state_type]:
                    hiddens.append(state[_l][_s])

        return analyze_util.flat_combine(hiddens)
