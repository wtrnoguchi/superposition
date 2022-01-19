import argparse
import os
import hashlib
import random
import subprocess

import h5py
import numpy

import constants
import creator
import util

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default')
    parser.add_argument('--display', action='store_true', default=False)
    args = parser.parse_args()

    seed = int(hashlib.md5(args.config.encode('utf-8')).hexdigest()[:8], 16)
    random.seed(seed)
    numpy.random.seed(seed)

    config = util.load_config('config/collect/{:s}.yml'.format(args.config))
    env = creator.create_environment(config.environment)

    save_dir = constants.SAVE_ROOT_DIR + args.config + '/'
    subprocess.check_output(['mkdir', '-p', save_dir])

    save_path = save_dir + constants.SAVE_DATA_NAME
    assert not os.path.isfile(save_path)

    h5_file = h5py.File(save_path, 'w')

    env.init()
    env.set_camera('self')
    if not args.display:
        env.off_display()

    if config.grid:
        n = 0

        xmin = env.world.config.xmin
        xmax = env.world.config.xmax
        ymin = env.world.config.ymin
        ymax = env.world.config.ymax

        N = config.num_div * config.num_div

        m_shape = (N, N, constants.MOTION_DIM)
        v_shape = (N, N, env.world.config.camera.agent.height,
                   env.world.config.camera.agent.width,
                   constants.VISION_CHANNELS)
        p_shape = m_shape

        mode = 'train'

        data = {
            'self_vision':
            h5_file.create_dataset(mode + '/self_vision', v_shape, dtype='f'),
            'other_vision':
            h5_file.create_dataset(mode + '/other_vision', v_shape, dtype='f'),
            'self_vision_no_agent':
            h5_file.create_dataset(
                mode + '/self_vision_no_agent', v_shape, dtype='f'),
            'other_vision_no_agent':
            h5_file.create_dataset(
                mode + '/other_vision_no_agent', v_shape, dtype='f'),
            'self_position':
            h5_file.create_dataset(
                mode + '/self_position', p_shape, dtype='f'),
            'other_position':
            h5_file.create_dataset(
                mode + '/other_position', p_shape, dtype='f'),
        }

        for ox in numpy.linspace(xmin, xmax, config.num_div):
            for oy in numpy.linspace(ymin, ymax, config.num_div):

                op = numpy.array([ox, oy])
                t = 0
                for sx in numpy.linspace(xmin, xmax, config.num_div):
                    for sy in numpy.linspace(ymin, ymax, config.num_div):
                        print(n, t)

                        sp = numpy.array([sx, sy])
                        env.set_agent_pos(sp, op)

                        env.set_camera('self')
                        self_vision = env.capture()

                        env.set_camera('other')
                        other_vision = env.capture()

                        env.set_camera('self')
                        env.world.set_visible(False, False)
                        self_vision_no_agent = env.capture()
                        env.set_camera('other')
                        env.world.set_visible(False, False)
                        other_vision_no_agent = env.capture()

                        data['self_vision'][n, t, :] = self_vision
                        data['other_vision'][n, t, :] = other_vision
                        data['self_vision_no_agent'][
                            n, t, :] = self_vision_no_agent
                        data['other_vision_no_agent'][
                            n, t, :] = other_vision_no_agent
                        data['self_position'][n, t, :] = sp
                        data['other_position'][n, t, :] = op

                        t += 1
                n += 1

    else:

        for mode, n_data in config.n_data.items():
            print(mode, n_data)
            m_shape = (n_data, config.seq_length, constants.MOTION_DIM)
            p_shape = m_shape
            v_shape = (n_data, config.seq_length,
                       env.world.config.camera.agent.height,
                       env.world.config.camera.agent.width,
                       constants.VISION_CHANNELS)
            data = {
                'self_motion':
                h5_file.create_dataset(
                    mode + '/self_motion', m_shape, dtype='f'),
                'self_vision':
                h5_file.create_dataset(
                    mode + '/self_vision', v_shape, dtype='f'),
                'self_position':
                h5_file.create_dataset(
                    mode + '/self_position', p_shape, dtype='f'),
                'other_motion':
                h5_file.create_dataset(
                    mode + '/other_motion', m_shape, dtype='f'),
                'other_position':
                h5_file.create_dataset(
                    mode + '/other_position', p_shape, dtype='f'),
            }
            if config.other_vision:
                data['other_vision'] = h5_file.create_dataset(
                    mode + '/other_vision', v_shape, dtype='f')
            # do collect
            for n in range(n_data):
                env.reset()

                for t in range(config.seq_length):
                    print(n, t)

                    if config.other_vision:
                        env.set_camera('other')
                        ov = env.capture()
                        data['other_vision'][n, t, :] = ov
                        env.set_camera('self')

                    sv, sm, om, sp, op = env.step()

                    data['self_vision'][n, t, :] = sv
                    data['self_motion'][n, t, :] = sm
                    data['other_motion'][n, t, :] = om
                    data['self_position'][n, t, :] = sp
                    data['other_position'][n, t, :] = op
