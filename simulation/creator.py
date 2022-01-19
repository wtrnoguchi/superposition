import os

try:
    from . import util
    from .environment import Environment
    from .simulation import World, agent
except Exception:
    import util
    from environment import Environment
    from simulation import World, agent

dirpath = os.path.dirname(os.path.abspath(__file__))


def create_environment(config=None):

    if config is None:
        config = util.load_config(dirpath +
                                  '/config/environment/default.yml')

    return Environment(config)


def create_world(config_name):

    config = util.load_config(dirpath +
                              '/config/world/{:s}.yml'.format(config_name))

    return World(config=config)


def create_agent(config_name):

    config = util.load_config(dirpath +
                              '/config/agent/{:s}.yml'.format(config_name))

    return getattr(agent, config.type)(config)
