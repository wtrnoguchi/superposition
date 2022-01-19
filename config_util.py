import yaml


def load_config(f_name, to_dict=True):

    with open(f_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if to_dict:
        return DotDict(config)
    else:
        return config


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
