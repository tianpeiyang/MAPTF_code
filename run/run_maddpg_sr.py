import time
import pickle
import tensorflow as tf
from util.output_json import OutputJson
import numpy as np
from alg.maddpg.common import tf_util as U
from alg.maddpg.trainer.maddpg import MADDPGAgentTrainer
from alg.muti_ptf_ppo.ppo import PPO
import tensorflow.contrib.layers as layers
from game.particle.make_env import make_env
from alg.option.option_ma_sro import SROption, SREmbedding
from alg.common import common


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def get_ppo_args():
    args = dict()
    args['policy'] = 'policy'
    args['old_policy'] = 'old_policy'
    args['c2'] = 0.001
    args['stochastic'] = True
    args['n_layer_a_1'] = 64
    args['n_layer_a_2'] = 64
    args['n_layer_c_1'] = 64
    args['n_layer_c_2'] = 64
    args['continuous_action'] = False
    args['optimizer'] = 'adam'
    args['learning_rate_a'] = 0.0003
    args['learning_rate_c'] = 0.0003
    args['clip_value'] = 0.2
    return args


def get_trainers(env, num_adversaries, obs_shape_n, args, SESS, logger):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    print(num_adversaries)
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n[0: num_adversaries], env.action_space[0: num_adversaries], i, args, use_option=args['adv_use_option'],
            local_q_func=(args['adv_policy'] == 'ddpg')))
    for i in range(num_adversaries, env.n):
        if args['good_load_model']:
            n_actions, n_features = env.action_space[num_adversaries].n, env.observation_space[num_adversaries].shape[0]
            ppo = PPO(n_actions, n_features, get_ppo_args(), SESS, logger, i)
            trainers.append(ppo)
        else:
            trainers.append(trainer(
                "agent_%d" % i, model, list(obs_shape_n[num_adversaries: env.n]), list(env.action_space[num_adversaries: env.n]), i, args, use_option=args['good_use_option'],
                local_q_func=(args['good_policy'] == 'ddpg')))
    return trainers


def get_options(env, args, SESS, logger):
    adv_option = None
    good_option = None
    # FIXME 1105 - init SR option and SR Embedding
    if args['adv_use_option']:
        if args['num_adversaries'] > 0:
            adv_emb = SREmbedding(args['num_adversaries'], args['num_adversaries'], env.observation_space[0].shape[0],
                                  args, SESS, logger, "adv", emb_dim=args['embedding_dim'])
            adv_option = SROption(args['num_adversaries'], args['num_adversaries'], env.observation_space[0].shape[0],
                                  args, SESS, logger, "adv", sr_emb=adv_emb)
    if args['good_use_option']:
        if env.n - args['num_adversaries'] > 0:
            good_emb = SREmbedding(env.n - args['num_adversaries'], env.n - args['num_adversaries'], env.observation_space[args['num_adversaries']].shape[0],
                                   args, SESS, logger, "good", emb_dim=args['embedding_dim'])
            good_option = SROption(env.n - args['num_adversaries'], env.n - args['num_adversaries'],
                                   env.observation_space[args['num_adversaries']].shape[0],
                                   args, SESS, logger, "good", sr_emb=good_emb)

    # FIXME 1105 - use the shared big SR model for agents at the same side
    option_list = []
    for i in range(env.n):
        if i < args['num_adversaries']:
            option_list.append(adv_option)
        else:
            option_list.append(good_option)
    return option_list


def run(args, env, alg, logger):
    with U.single_threaded_session():
        # Create environment
        env = make_env(args)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range (env.n)]
        num_adversaries = min(env.n, args['num_adversaries'])
        trainers = get_trainers(env, args['num_adversaries'], obs_shape_n, args, U.get_session(), logger)
        print('Using good policy {} and adv policy {}'.format(args['good_policy'], args['adv_policy']))
        option_list = get_options(env, args, U.get_session(), logger)
        # Initialize
        U.initialize()

        N_O = env.n

        for n in range(num_adversaries, env.n):
            if args['good_load_model']:
                trainers[n].load_model(args['good_load_model_path'] + '_' + str(n))

        field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean', 'episode']
        OJ = OutputJson(field)
        total_step = 0

        memory = []
        [memory.append(np.repeat(0, env.n, axis=0)) for i in range(args['reward_memory'])]
        memory = np.array(memory)

        discount_memory = []
        [discount_memory.append(np.repeat(0, env.n, axis=0)) for i in range(args['reward_memory'])]
        discount_memory = np.array(discount_memory)

        memory_count = 0

        totalreward = np.zeros(args['numGames'])
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        t_start = time.time()

        print('Starting iterations...')
        for episode in range(args['numGames']):
            step = 0
            episode_reward = np.zeros(env.n)
            episode_discount_reward = np.zeros(env.n)
            obs_n = env.reset()
            done_epi = [False for i in range(env.n)]
            opt_n = np.zeros(N_O, dtype=np.int)
            term_n = np.repeat(1, N_O)
            while True:
                for n in range(N_O):
                    if n < args['num_adversaries']:
                        if np.random.uniform() < term_n[n]:
                            opt_n[n] = option_list[n].choose_o(obs_n[n], agent_id=n) if option_list[
                                                                                                 n] is not None else 0
                        term_n[n] = option_list[n].get_t(obs_n[n], opt_n[n], agent_id=n) if option_list[
                                                                                                     n] is not None else 0
                    else:
                        if np.random.uniform() < term_n[n]:
                            opt_n[n] = option_list[n].choose_o(obs_n[n], agent_id=n - args['num_adversaries']) if \
                            option_list[n] is not None else 0
                        term_n[n] = option_list[n].get_t(obs_n[n], opt_n[n],
                                                         agent_id=n - args['num_adversaries']) if option_list[
                                                                                                      n] is not None else 0

                # get action
                action_n = []
                for n in range(num_adversaries):
                    if args['is_soft_max_action']:
                        action_n.append(trainers[n].action(obs_n[n], trainers[int(opt_n[n])], term_n[n], episode))
                    else:
                        action_n.append(trainers[n].action(obs_n[n]))
                for n in range(num_adversaries, N_O):
                    if args['good_load_model']:
                        action_n.append(trainers[n].choose_action(obs_n[n]))
                    else:
                        if args['is_soft_max_action']:
                            action_n.append(trainers[n].action(obs_n[n], trainers[int(opt_n[n])], term_n[n], episode))
                        else:
                            action_n.append(trainers[n].action(obs_n[n]))
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(np.copy(action_n))
                reward_n = rew_n
                #print(info_n)
                done = all(done_n)
                terminal = (step >= args['epi_step'])

                opa_n = [[] for i in range(N_O)]
                if args['adv_use_option']:
                    if args['other_option_update']:
                        adv_actions = []
                        adv_obs = obs_n[0: args['num_adversaries']]
                        adv_obs = [list(temp) for temp in adv_obs]
                        for n in range(args['num_adversaries']):
                            adv_actions.append(trainers[n].choose_deterministic_action(adv_obs))
                        for n in range(args['num_adversaries']):
                            opa = []
                            for i in range(args['num_adversaries']):
                                if i == opt_n[n] or (
                                common.action_equal(adv_actions[i][n], action_n[n], args['continuous_action'])):
                                    opa.append(1)
                                else:
                                    opa.append(0)
                            opa_n[n] = opa
                    else:
                        for n in range(args['num_adversaries']):
                            opa = []
                            for i in range(args['num_adversaries']):
                                if i == opt_n[n]:
                                    opa.append(1)
                                else:
                                    opa.append(0)
                            opa_n[n] = opa
                if args['good_use_option']:
                    if args['other_option_update']:
                        good_actions = []
                        good_obs = obs_n[args['num_adversaries']: N_O]
                        good_obs = [list(temp) for temp in good_obs]
                        for n in range(args['num_adversaries'], N_O):
                            good_actions.append(trainers[n].choose_deterministic_action(good_obs))
                        for n in range(N_O - args['num_adversaries']):
                            opa = []
                            for i in range(N_O - args['num_adversaries']):
                                if i == opt_n[n + args['num_adversaries']] or (
                                common.action_equal(good_actions[i][n], action_n[n + args['num_adversaries']],
                                                    args['continuous_action'])):
                                    opa.append(1)
                                else:
                                    opa.append(0)
                            opa_n[n + args['num_adversaries']] = opa
                    else:
                        for n in range(N_O - args['num_adversaries']):
                            opa = []
                            for i in range(N_O - args['num_adversaries']):
                                if i == opt_n[n + args['num_adversaries']]:
                                    opa.append(1)
                                else:
                                    opa.append(0)
                            opa_n[n + args['num_adversaries']] = opa


                # collect experience
                for i, agent in enumerate(trainers):
                    if i >= num_adversaries and args['good_load_model']:
                        continue
                    agent.experience(obs_n[i], action_n[i], reward_n[i], new_obs_n[i], opt_n[i], term_n[i], done_n[i], terminal)
                obs_n = new_obs_n

                for i in range(env.n):
                    if option_list[i] is not None and not done_n[i]:
                        if i < args['num_adversaries']:
                            option_list[i].store_transition(obs_n[i], action_n[i], rew_n[i], done_n[i],
                                                        new_obs_n[i], opa_n[i], agent_id=i)
                        else:
                            option_list[i].store_transition(obs_n[i], action_n[i], rew_n[i], done_n[i],
                                                        new_obs_n[i], opa_n[i], agent_id=i - args['num_adversaries'])
                    if total_step > args['learning_step']:
                        if option_list[i] is not None and not done_n[i]:
                            if i < args['num_adversaries']:
                                actor = trainers[0: args['num_adversaries']]
                            else:
                                actor = trainers[args['num_adversaries']: env.n]
                            if i < args['num_adversaries']:
                                option_list[i].update(obs_n[i], opt_n[i], reward_n[i], done_n[i], new_obs_n[i],
                                                  actor, agent_id=i)
                            else:
                                option_list[i].update(obs_n[i], opt_n[i], reward_n[i], done_n[i], new_obs_n[i],
                                                      actor, agent_id=i - args['num_adversaries'])

                # update all trainers, if not in display or benchmark mode
                loss = None
                if not args['adv_load_model']:
                    for i, agent in enumerate(trainers[0: num_adversaries]):
                        loss = agent.update(trainers[0: num_adversaries], total_step, episode - args['batch_size'])
                        if loss is not None:
                            logger.write_tb_log('q_loss', loss[0], episode)
                            logger.write_tb_log('loss_cr', loss[1], episode)
                            logger.write_tb_log('p_loss', loss[2], episode)
                            logger.write_tb_log('cross_entropy', loss[3], episode)
                if not args['good_load_model']:
                    for i, agent in enumerate(trainers[num_adversaries: env.n]):
                        loss = agent.update(trainers[num_adversaries: env.n], total_step, episode - args['batch_size'])
                        if loss is not None:
                            logger.write_tb_log('q_loss', loss[0], episode)
                            logger.write_tb_log('loss_cr', loss[1], episode)
                            logger.write_tb_log('p_loss', loss[2], episode)
                            logger.write_tb_log('cross_entropy', loss[3], episode)

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                reward = np.array(rew_n)
                episode_discount_reward = [episode_discount_reward[i] + reward[i] * np.power(args['reward_decay'], step) for i in range(env.n)]
                episode_reward = [episode_reward[i] + reward[i] for i in range(env.n)]

                if done or terminal:
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])
                    # print(episode_rewards[-2])
                    #episode_discount_reward = episode_discount_reward + round(episode_rewards[-2] * np.power(args['reward_decay'], step), 8)
                    discount_memory[episode % args['reward_memory']] = episode_discount_reward
                    memory[episode % args['reward_memory']] = episode_reward
                    totalreward[episode] = episode_rewards[-2]
                    mean_memory = np.mean(memory, axis=0)
                    discount_mean_memory = np.mean(discount_memory, axis=0)
                    OJ.update([done_n, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory,
                              episode])
                    OJ.print_first()

                    logger.write_tb_log('discount_reward', episode_discount_reward, episode)
                    logger.write_tb_log('discount_reward_mean', discount_mean_memory, episode)
                    logger.write_tb_log('undiscounted_reward', episode_reward, episode)
                    logger.write_tb_log('reward_mean', mean_memory, episode)

                    for i in range(env.n):
                        if total_step > args['learning_step']:
                            if option_list[i] is not None:
                                option_list[i].update_e()

                    break

                # increment global step counter
                step += 1
                total_step += 1
                '''
                # for benchmarking learned policies
                if args['benchmark']:
                    for i, info in enumerate(info_n):
                        agent_info[-1][i].append(info_n['n'])
                    if total_step > args['benchmark_iters'] and (done or terminal):
                        file_name = args['results_path'] + args['benchmark_dir'] + '/benchmark.pkl'
                        print('Finished benchmarking, now saving...')
                        with open(file_name, 'wb') as fp:
                            pickle.dump(agent_info[:-1], fp)
                        break
                    continue

                # for displaying learned policies
                if args['display']:
                    time.sleep(0.1)
                    env.render()
                    continue
                '''
            # save model, display training output
            if terminal and (len(episode_rewards) % args['save_per_episodes'] == 0):
                U.save_state(args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(episode), saver=saver)
                OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        total_step, len(episode_rewards), np.mean(episode_rewards[-args['save_per_episodes']:]),
                        round(time.time() - t_start, 3)))
                else:
                    print(
                        "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format (
                            total_step, len(episode_rewards), np.mean(episode_rewards[-args['save_per_episodes']:]),
                            [np.mean(rew[-args['save_per_episodes']:]) for rew in agent_rewards],
                            round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-args['save_per_episodes']:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-args['save_per_episodes']:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > args['numGames']:
                rew_file_name = args['results_path'] + args['reward_output'] + '/rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = args['results_path'] + args['reward_output'] + '/agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))

        OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])
        if args['save_model']:
            U.save_state(args['results_path'] + args['SAVE_PATH'] + "/model", saver=saver)
