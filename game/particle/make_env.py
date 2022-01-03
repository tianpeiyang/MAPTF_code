"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
import numpy as np
import time


def make_env(args):
    scenario_name = args['game_name']
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from game.particle.multiagent.environment import MultiAgentEnv
    import game.particle.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    if args['benchmark'] and not args['obs_sort']:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, args=args)
    elif not args['benchmark'] and args['obs_sort']:
        if args["reward_func"] == "reward2":
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward2, scenario.observation, scenario.observation_sort, scenario.is_done2, args=args)
        elif args["reward_func"] == "reward3":
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward3, scenario.observation3, scenario.observation_sort3, scenario.is_done3, args=args)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                scenario.observation_sort, scenario.is_done, args=args)
    elif not args['benchmark'] and not args['obs_sort']:
        if args["reward_func"] == "reward2":
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward2, scenario.observation, args=args)
        elif args["reward_func"] == "reward3":
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward3, scenario.observation3, args=args)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, args=args)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward2, scenario.observation, args=args)
    return env


# test
def action(obs):
    if env.discrete_action_space:
        i = np.random.randint(0, 5)
        u = np.zeros(5)
        u[i] = 1
    else:
        u = np.array([(np.random.random() - 0.5) * 2, (np.random.random() - 0.5) * 2])
    return u


if __name__ == '__main__':
    args = dict()
    args['game_name'] = "simple_spread_old"
    args['benchmark'] = False
    args['obs_sort'] = False
    args['reward_func'] = 'reward'
    args['restrict_move'] = True
    args['num_adversaries'] = 0
    args['num_good'] = 6
    env = make_env(args)
    print(env.action_space)
    env.render()
    # create interactive policies for each agent
    # execution loop
    obs_n = env.reset()
    print(env.observation_space)
    print(env.action_space)

    for ep in range(100):
        obs_n = env.reset()
        step = 0
        reward = np.zeros(env.n)
        done = [False for i in range(env.n)]
        while True:
            # query for action from each agent's policy
            act_n = []
            for i in range(env.n):
                act_n.append(action(obs_n[i]))
            #print(act_n)
            #print(act_n)
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            for i in range(env.n):
                if not done[i]:
                    done[i] = done_n[i]
            reward += reward_n
            #print(obs_n)
            # render all agent views
            #time.sleep(0.1)
            env.render()
            step += 1
            if step > 100 or all((done_n[i] is True for i in range(env.n))):
                print(step, reward, done_n)
                break


