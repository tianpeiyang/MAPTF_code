import numpy as np
import os
import yaml
import gym.spaces
import sys
import tensorflow as tf
from gym.utils import seeding
import random

from alg import REGISTRY as alg_REGISTRY
from game import REGISTRY as env_REGISTRY
from run import REGISTRY as run_REGISTRY
from util.logger import Logger
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def default(str):
    return str + ' [Default: %default]'


def config_args(config_name):
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", "{}.yaml".format(config_name)), "r") as f:
            try:
                #config_dict = yaml.load(f, Loader=yaml.FullLoader)
                config_dict = yaml.load(f)
                return config_dict
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)


def readCommand(argv):
    """
    Processes the command used to run main from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python main.py <options>
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=20000)
    parser.add_option('-e', '--epi_step', dest='epi_step', type='int',
                      help=default('the steps of each episode'), default=99)
    parser.add_option('-g', '--game', dest='game',
                      help=default('use which GAME to play'), default='pacman')
    parser.add_option('-a', '--alg', dest='algorithm',
                      help=default('use which algorithm to play'), default='multi_ppo')
    parser.add_option('-c', '--alg_conf', dest='algorithm_config',
                      help=default('algorithm config'), default='ppo_conf.yaml')
    parser.add_option('-d', '--env_conf', dest='environment_config',
                      help=default('Environment config'), default='pacman_conf')
    parser.add_option('-s', '--seed', dest='seed', type='int',
                      help=default('the seed of tf'), default=1234)
    parser.add_option('-o', '--optimizer', dest='optimizer',
                      help=default('the optimizer of tensorflow'), default='adam')
    parser.add_option('-t', '--run_test', dest='run_test',
                      help=default('run test'), default=False)

    """
    parser.add_option('-f', '--fileName', dest='fileName',
                      help=default('the file name'), default='dqn_pinball')
    parser.add_option('-m', '--modelName', dest='modelName',
                      help=default('the model name'), default='dqn_pinball')
    """

    options, otherjunk = parser.parse_args(argv)
    # print(type(options))

    alg_conf = options.algorithm_config
    env_conf = options.environment_config
    alg_config_dict = config_args(alg_conf)
    env_config_dict = config_args(env_conf)

    args = dict()
    args['numGames'] = options.numGames
    args['game'] = options.game
    args['algorithm'] = options.algorithm
    args['epi_step'] = options.epi_step
    args['seed'] = options.seed
    args['optimizer'] = options.optimizer
    args['run_test'] = options.run_test
    t = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    if alg_config_dict is not None:
        args = dict(args, **alg_config_dict)
    if env_config_dict is not None:
        args = dict(args, **env_config_dict)

    #args['fileName'] = options.fileName
    #args['optimizer'] = options.optimizer

    for item in otherjunk:
        key = item.split('=')[0]
        value = item.split('=')[1]
        #print(key, value)
        if key not in args:
            raise Exception('Command line input not understood: ' + str(item))
        if type(args[key]) is int:
            args[key] = int(value)
        elif type(args[key]) is float:
            args[key] = float(value)
        elif type(args[key]) is str:
            args[key] = str(value)
        elif type(args[key]) is bool:
            if str(value).lower() == 'true':
                args[key] = True
            elif str(value).lower() == 'false':
                args[key] = False
            else:
                raise Exception('Command line input is not boolean type: ' + str(value))
        elif type(args[key]) is list:
            try:
                args[key] = eval(value)
            except (SyntaxError, NameError):
                value_l = str(value).replace(' ', '').replace('[', '').replace(']', '').split(',')
                args[key] = value_l
        else:
            raise Exception('Command line input is not valid type: ' + str(value))

    args['results_path'] = "../results/" + args['algorithm'] + "/" + args['game'] + "/" + args[
        'game_name'] + "/" + t + "/"

    if not args['run_test']:
        if not os.path.exists(args['results_path']):
            os.makedirs(args['results_path'])
        if not os.path.exists(args['results_path'] + args['SAVE_PATH']):
            os.makedirs(args['results_path'] + args['SAVE_PATH'])
        if not os.path.exists(args['results_path'] + args['graph_path']):
            os.makedirs(args['results_path'] + args['graph_path'])
        if not os.path.exists(args['results_path'] + args['reward_output']):
            os.makedirs(args['results_path'] + args['reward_output'])
        if not os.path.exists(args['results_path'] + args['log']):
            os.makedirs(args['results_path'] + args['log'])

        with open(
                args['results_path'] + "command.txt",
                'w') as f:
            out = ' '.join(argv)
            f.writelines(out)

        with open(args['results_path'] + "args.json", "w") as f:
            json.dump(args, f)

        # print('args', args)

    return args


def get_space(env):
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        action_dim = env.action_space.n
    elif type(env.action_space) is gym.spaces.box.Box:
        action_dim = env.action_space.shape[0]
    elif type(env.action_space) is int:
        action_dim = env.action_space
    elif type(env.action_space) is list:
        if type(env.action_space[0]) is gym.spaces.box.Box:
            action_dim = env.action_space[0].shape[0]
        else:
            action_dim = env.action_space[0].n
    else:
        raise Exception('action space is not a valid '
                        '.type')
    if type(env.observation_space) is gym.spaces.discrete.Discrete:
        features = env.observation_space.n
    elif type(env.observation_space) is gym.spaces.box.Box:
        features = env.observation_space.shape[0]
    elif type(env.observation_space) is int:
        features = env.observation_space
    elif type(env.observation_space) is list:
        features = env.observation_space[0].shape[0]
    else:
        raise Exception('observation space is not a valid type')
    return action_dim, features

def NoneAlg(alg):
    algs = ['maddpg', 'multi_ppo', 'multi_ppo_sro', 'maddpg_sr' , 'shppo', 'shppo_sro']
    if alg in algs:
        return True
    return False


def runGames(args):
    print(args)
    if args['run_test']:
        logger = None
    else:
        logger = Logger(args['results_path'] + args['log'], args['results_path'] + args['graph_path'], args)
    np.random.seed(args['seed'])
    tf.set_random_seed(args['seed'])
    random.seed(args['seed'])
    seeding.np_random(args['seed'])
    env = env_REGISTRY[args['game']](args)
    args['action_dim'], args['features'] = get_space(env)
    if NoneAlg(args['algorithm']):
        alg = None
    else:
        alg = alg_REGISTRY[args['algorithm']](args['action_dim'], args['features'], args, logger)
    if args['run_test'] and args['game'] != 'particle':
        run_REGISTRY['test'](args, env, alg, logger)
    elif args['run_test'] and args['game'] == 'particle':
        run_REGISTRY['particle'](args, env, alg, logger)
    else:
        run_REGISTRY[args['algorithm']](args, env, alg, logger)


if __name__ == '__main__':
    args = readCommand(sys.argv[1:])  # Get game components based on input
    runGames(args)







