import argparse

import torch
import torch.optim as optim

import args_util
import model as models
from exp import runner as runners
import util

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args_util.add_common_args(parser)
    args = parser.parse_args()

    util.set_cudnn_config(args)

    util.seed_all(args.seed)

    exp_config = util.gen_exp_config(args)
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(args, test=False)

    device = torch.device(args.device)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(device)

    util.load_pretrain(model, exp_config)

    params = util.gen_optim_params(model, exp_config.train.optim_config)

    optimizer = getattr(optim, exp_config.train.optimizer)(params)

    runner = getattr(runners, exp_config.runner)()

    train_loader, eval_loader, test_loader = util.gen_data_loader(
        exp_config.data.name, exp_config.train.batch_size,
        args.data_load_memory, device)

    train_logger, eval_logger, test_logger = util.gen_logger(log_dir)

    util.save_model_optimizer(model_dir, 0, model, optimizer)
    for epoch in range(1, exp_config.train.max_epochs + 1):

        runner.train_epoch(
            model,
            optimizer,
            epoch,
            exp_config,
            train_loader,
            train_logger,
        )

        if epoch % exp_config.train.save_interval == 0:
            util.save_model_optimizer(model_dir, epoch, model, optimizer)

        if epoch % exp_config.train.test_interval == 0:
            with torch.no_grad():

                runner.eval_epoch(
                    model,
                    epoch,
                    exp_config,
                    eval_loader,
                    eval_logger,
                )

                runner.eval_epoch(
                    model,
                    epoch,
                    exp_config,
                    test_loader,
                    test_logger,
                )
