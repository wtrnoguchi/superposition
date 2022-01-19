import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument(
        '--targets',
        nargs='*',
        type=str,
        default=['input', 'prediction', 'truth', 'reconstruction'])
    parser.add_argument('--img_ext', default='png')
    args = parser.parse_args()

    save_dir = 'gif/'

    subprocess.check_output(['mkdir', '-p', save_dir])

    for t in args.targets:

        imgs_path = '{:s}/{:05d}_*.{:s}'.format(t, args.idx, args.img_ext)
        save_name = '{:05d}_{:s}.gif'.format(args.idx, t)
        cmd = ['convert', '-delay', '5', imgs_path, save_dir + save_name]
        try:
            subprocess.check_output(cmd)
        except Exception:
            print(t, 'does not exist')
