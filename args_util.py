def add_common_args(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp_config')
    parser.add_argument(
        '--cudnn_deterministic', action='store_true', default=False)
    parser.add_argument(
        '--data_load_memory', action='store_true', default=False)


def add_test_args(parser):
    parser.add_argument('--test_epoch', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--test_data_name')
    parser.add_argument('--save_targets', nargs='*', default=[])
    parser.add_argument(
        '--save_targets_index', nargs='*', type=int, default=[])
    parser.add_argument(
        '--test_modes', nargs='*', type=str, default=['eval', 'test'])
    parser.add_argument('--img_ext', default='jpg')
    parser.add_argument(
        '--save_vision_img', action='store_true', default=False)
    parser.add_argument(
        '--mask_off', action='store_true', default=False)
