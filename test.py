import argparse

import args_util
import model as models
import torch
import util
from exp import runner as runners
from exp import saver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args_util.add_common_args(parser)
    args_util.add_test_args(parser)
    args = parser.parse_args()

    util.set_cudnn_config(args)

    util.seed_all(args.seed)

    exp_config = util.gen_exp_config(args)
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir, save_dir = util.gen_dirs(
        args, test=True, test_name=args.test_data_name)

    if args.mask_off:
        exp_config.p_mask_vision = 0

    device = torch.device(args.device)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(device)

    util.load_model(model_dir, args.test_epoch, model)

    runner = getattr(runners, exp_config.runner)()

    _, eval_loader, test_loader = util.gen_data_loader(
        args.test_data_name, args.test_batch_size, args.data_load_memory,
        device)

    train_logger, eval_logger, test_logger = util.gen_logger(log_dir)

    eval_saver = saver.DataSaver(
        save_dir=save_dir,
        mode='eval',
        save_targets=args.save_targets,
        save_targets_index=args.save_targets_index,
        save_vision_img=args.save_vision_img,
        img_ext=args.img_ext,
    )
    test_saver = saver.DataSaver(
        save_dir=save_dir,
        mode='test',
        save_targets=args.save_targets,
        save_targets_index=args.save_targets_index,
        save_vision_img=args.save_vision_img,
        img_ext=args.img_ext,
    )

    model.eval()

    with torch.no_grad():

        if 'eval' in args.test_modes:
            runner.eval_epoch(
                model,
                args.test_epoch,
                exp_config,
                eval_loader,
                eval_logger,
                eval_saver,
            )

        if 'test' in args.test_modes:
            runner.eval_epoch(
                model,
                args.test_epoch,
                exp_config,
                test_loader,
                test_logger,
                test_saver,
            )
