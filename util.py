import random
import subprocess
import sys

import h5py
import numpy

import torch
from config_util import load_config
from constants import (DATASET_DIR, DATASET_NAME, EXP_CONFIG_DIR, LOG_DIR_BASE,
                       MODEL_CONFIG_DIR, MODEL_DIR_BASE, RESULT_DIR,
                       SAVE_DIR_BASE, TEST_DIR_BASE)
from exp import loader, logger
import util


def gen_data_loader(data_name, batch_size, data_load_memory, device):
    data_filename = gen_data_filename(data_name)
    data = load_data(data_filename, data_load_memory)
    train_loader = loader.DataLoader(
        data['train'], batch_size, shuffle=True, device=device)

    eval_loader = loader.DataLoader(
        data['train'], batch_size, shuffle=False, device=device)

    if not ('test' in data.keys()):
        data['test'] = data['train']

    test_loader = loader.DataLoader(
        data['test'], batch_size, shuffle=False, device=device)

    return train_loader, eval_loader, test_loader


def gen_logger(log_dir):

    train_logger = logger.Logger(log_dir, 'train')
    eval_logger = logger.Logger(log_dir, 'eval')
    test_logger = logger.Logger(log_dir, 'test')
    return train_logger, eval_logger, test_logger


def load_model(model_dir, restore_epoch, model):
    model.load_state_dict(
        torch.load(model_dir + '{:05d}.pth'.format(restore_epoch))['model'])


def load_pretrain(model, exp_config):
    if not (exp_config.train.pretrain is None):
        model.load_state_dict(
            torch.load(
                gen_model_dir(gen_result_dir(exp_config.train.pretrain)) +
                '{:05d}.pth'.format(exp_config.train.pretrain.epoch))['model'],
            strict=exp_config.load_state_dict_strict)


def save_model_optimizer(model_dir, epoch, model, optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, model_dir + '{:05d}.pth'.format(epoch))


def mkdir_p(dir_path):
    subprocess.check_output(['mkdir', '-p', dir_path])


def r_print(string):
    sys.stdout.write("\r%s " % string)
    sys.stdout.flush()


def gen_data_filename(data_name):
    return DATASET_DIR + '{:s}/{:s}'.format(data_name, DATASET_NAME)


def gen_exp_config(args):
    return load_config(EXP_CONFIG_DIR + '{:s}.yml'.format(args.exp_config))


def gen_model_config(args):
    return load_config(MODEL_CONFIG_DIR + '/{:s}/{:s}.yml'.format(
        args.model.name, args.model.config))


def gen_dirs(args, test, test_name=None):
    result_dir = gen_result_dir(args)
    model_dir = gen_model_dir(result_dir)
    if test:
        assert (test_name is not None)
        test_dir = gen_test_dir(result_dir, test_name)
        log_dir = gen_log_dir(test_dir)
        save_dir = gen_save_dir(test_dir)
        return result_dir, model_dir, log_dir, save_dir
    else:
        log_dir = gen_log_dir(result_dir)
        return result_dir, model_dir, log_dir


def gen_result_dir(args):
    return RESULT_DIR + '{:s}/{:d}/'.format(args.exp_config, args.seed)


def gen_model_dir(result_dir):
    model_dir = result_dir + MODEL_DIR_BASE
    mkdir_p(model_dir)
    return model_dir


def gen_log_dir(result_dir):
    return result_dir + LOG_DIR_BASE


def gen_save_dir(result_dir):
    return result_dir + SAVE_DIR_BASE


def gen_test_dir(result_dir, test_name):
    return result_dir + TEST_DIR_BASE + test_name + '/'


def set_cudnn_config(args):
    torch.backends.cudnn.enabled = True
    if args.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gen_optim_params(net, config):
    train_params = {}
    freezed_params = {}

    for name, param in net.named_parameters():
        param.grad = None

        if False if config.freeze is None else any(
                s in name for s in config.freeze):
            param.requres_grad = False
            freezed_params[name] = param
        else:
            param.requres_grad = True
            train_params[name] = param

    default = {'params': []}
    default.update(config.default)

    params = {}

    for pp in config.per_params:
        params[pp['name']] = {'params': []}
        params[pp['name']].update(pp['args'])

    for name, param in train_params.items():
        param.requires_grad = True

        # assert (sum([s in name for s in config.per_params.keys()]) <= 1)

        if any(pp['name'] in name for pp in config.per_params):

            for pp in config.per_params:
                if pp['name'] in name:
                    params[pp['name']]['params'].append(param)
                    # print(name, pp['name'])
                    break
        else:
            default['params'].append(param)

    print('Train params')
    for name in train_params.keys():
        print(name)
    print('')
    print('Freezed params')
    for name in freezed_params.keys():
        print(name)
    print('')

    params = list(params.values())
    params.append(default)

    return params


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transpose_vision(v, dim=5):
    if dim == 5:
        return v.transpose(0, 1, 4, 2, 3)
    elif dim == 4:
        return v.transpose(0, 3, 1, 2)
    elif dim == 3:
        return v.transpose(2, 0, 1)
    else:
        assert (False)


def de_transpose_vision(v, dim=5):

    if dim == 5:
        return v.transpose(0, 1, 3, 4, 2)
    elif dim == 4:
        return v.transpose(0, 2, 3, 1)
    elif dim == 3:
        return v.transpose(1, 2, 0)
    else:
        assert (False)


def scale_vision(v):
    return v * 2 - 1


def de_scale_vision(v):
    return (v + 1) / 2


def load_data(f_data, load_memory=False):
    f = h5py.File(f_data, 'r')

    def _recursive_load(loaded, source):

        for k, v in source.items():
            if isinstance(v, dict) or isinstance(v, h5py._hl.group.Group):
                loaded[k] = {}
                _recursive_load(loaded[k], v)
            else:
                loaded[k] = v[()] if load_memory else v

        return loaded

    data = _recursive_load({}, f)

    return data
