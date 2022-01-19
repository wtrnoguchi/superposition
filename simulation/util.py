import os

import yaml


def load_config(f_name, to_dict=True):

    config = load_yaml(f_name)

    if 'base' in config.keys():
        dirpath = os.path.dirname(f_name)
        base = load_config(dirpath + '/' + config.pop('base') + '.yml')
        base.update(config)
        config = base

    if to_dict:
        return DotDict(config)
    else:
        return config


def load_yaml(f_name):
    with open(f_name, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)

    return loaded


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [
                    DotDict(v) if isinstance(v, dict) else v for v in value
                ]
            self[key] = value
