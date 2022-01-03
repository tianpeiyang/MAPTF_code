import numpy as np
from util.output_json import OutputJson
from alg.muti_ptf_ppo.ppo_add_entropy import PPO as PPO_add_entropy
from alg.muti_ptf_ppo.ppo import PPO
from alg.option.option_ma_sro import SROption, SREmbedding
from alg.common.common import build_source_actor
import tensorflow as tf
from alg.common import common
import os

import time


def run(args, env, RL, logger):
    start = time.time()
    if args['use_gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpu_id']
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        SESS = tf.Session(config=config)
    else:
        SESS = tf.Session()

    N_O = env.n
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
            good_option = SROption(env.n - args['num_adversaries'], env.n - args['num_adversaries'], env.observation_space[args['num_adversaries']].shape[0],
                                   args, SESS, logger, "good", sr_emb=good_emb)

    # FIXME 1105 - use the shared big SR model for agents at the same side
    option_list = []
    for n in range(N_O):
        if n < args['num_adversaries']:
            option_list.append(adv_option)
        else:
            option_list.append(good_option)

    ppo_list = []
    for n in range(N_O):
        n_actions, n_features = env.action_space[n].n, env.observation_space[n].shape[0]
        # print(n_actions, n_features)
        if n < args['num_adversaries']:
            if args['adv_policy'] == "ppo" and args['adv_use_option']:
                ppo = PPO_add_entropy(n_actions, n_features, args, SESS, logger, n)
                ppo_list.append(ppo)
            elif args['adv_policy'] == "ppo" and not args['adv_use_option']:
                if args['adv_load_model']:  # 参数和model一致
                    load_args = {}
                    load_args['policy'] = args['adv_policy']
                    load_args['action_dim'] = n_actions
                    load_args['features'] = n_features
                    ppo = build_source_actor(load_args, SESS, args['adv_load_model_path'], n)
                else:
                    ppo = PPO(n_actions, n_features, args, SESS, logger, n)
                ppo_list.append(ppo)
        else:
            if args['good_policy'] == "ppo" and args['good_use_option']:
                ppo = PPO_add_entropy(n_actions, n_features, args, SESS, logger, n)
                ppo_list.append(ppo)
            elif args['good_policy'] == "ppo" and not args['good_use_option']:
                if args['good_load_model']:  # 参数和model一致
                    load_args = {}
                    load_args['policy'] = args['good_policy']
                    load_args['action_dim'] = n_actions
                    load_args['features'] = n_features
                    ppo = build_source_actor(load_args, SESS, args['good_load_model_path'], n)
                else:
                    ppo = PPO(n_actions, n_features, args, SESS, logger, n)
                ppo_list.append(ppo)

    SESS.run(tf.global_variables_initializer())

    for n in range(N_O):
        if n < args['num_adversaries']:
            if args['adv_load_model']:
                ppo_list[n].load_model(args['adv_load_model_path'] + '_' + str(n))
        else:
            if args['good_load_model']:
                ppo_list[n].load_model(args['good_load_model_path'] + '_' + str(n))

    if args['reload_model']:
        if args['num_adversaries'] == 0:
            for i in range(N_O - 1):
                ppo_list[i].load_model(args['reload_model_path'] + '_' + str(i))
        else:
            for i in range(args['num_adversaries']):
                ppo_list[i].load_model(args['reload_model_path'] + '_' + str(i))
        if good_option is not None:
            path = args['reload_model_path'].replace('model', 'good_option')
            good_option.load_model(path)
            good_option.epsilon = args['e_greedy']
        if adv_option is not None:
            path = args['reload_model_path'].replace('model_8000', 'adv_option_8000')
            adv_option.load_model(path)
            adv_option.epsilon = args['e_greedy']

    total_step = 0
    memory = np.zeros((args['reward_memory'], env.n))
    reward_memory_size = 0
    discount_memory = np.zeros((args['reward_memory'], env.n))
    numGames = args['numGames']
    field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean','episode']
    OJ = OutputJson(field)

    init = time.time()
    print("init: %.3f s" % (init-start))

    for episode in range(numGames):
        # initial observation
        observation = env.reset()
        # print(len(observation[0]))
        option_obs = np.array(observation)
        # print(observation)
        opt_n = np.zeros(N_O, dtype=np.int)
        term_n = np.repeat(1, N_O)

        step = 0
        episode_reward = np.zeros(env.n)
        episode_discount_reward = np.zeros(env.n)
        buffer_s, buffer_a, buffer_r, buffer_o, buffer_t, buffer_d = [], [], [], [], [], []
        done_n = [False for i in range(N_O)]
        ppo_done = [False for i in range(N_O)]

        action_time = 0
        step_time = 0
        opa_time = 0
        update_time = 0
        while True:
            act_n = []
            game_acts = []

            start = time.time()
            for n in range(N_O):
                if n < args['num_adversaries']:
                    if np.random.uniform() < term_n[n]:
                        opt_n[n] = option_list[n].choose_o(option_obs[n], agent_id=n) if option_list[n] is not None else 0
                    term_n[n] = option_list[n].get_t(option_obs[n], opt_n[n], agent_id=n) if option_list[n] is not None else 0
                else:
                    if np.random.uniform() < term_n[n]:
                        opt_n[n] = option_list[n].choose_o(option_obs[n], agent_id=n - args['num_adversaries']) if option_list[n] is not None else 0
                    term_n[n] = option_list[n].get_t(option_obs[n], opt_n[n], agent_id=n - args['num_adversaries']) if option_list[n] is not None else 0

            for n in range(N_O):
                action = ppo_list[n].choose_action(observation[n])
                act_n.append(action)
                game_acts.append(action)

            end = time.time()
            action_time += end - start
            # FIXME 1106 - to check
            # observation_, reward, done, _ = env.step(game_acts)
            start = time.time()
            observation_, reward, done, _ = env.step(np.copy(game_acts), done_n)
            end = time.time()
            step_time += end - start

            option_obs_ = observation_

            #print(done)
            start = time.time()
            opa_n = [[] for i in range(N_O)]
            if args['adv_use_option']:
                if args['other_option_update']:
                    adv_actions = []
                    adv_obs = observation[0: args['num_adversaries']]
                    adv_obs = [list(temp) for temp in adv_obs]
                    for n in range(args['num_adversaries']):
                        adv_actions.append(ppo_list[n].choose_deterministic_action(adv_obs))
                    for n in range(args['num_adversaries']):
                        opa = []
                        for i in range(args['num_adversaries']):
                            if i == opt_n[n] or (
                            common.action_equal(adv_actions[i][n], act_n[n], args['continuous_action'])):
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
                    good_obs = observation[args['num_adversaries']: N_O]
                    good_obs = [list(temp) for temp in good_obs]
                    for n in range(args['num_adversaries'], N_O):
                        good_actions.append(ppo_list[n].choose_deterministic_action(good_obs))
                    for n in range(N_O - args['num_adversaries']):
                        opa = []
                        for i in range(N_O - args['num_adversaries']):
                            if i == opt_n[n + args['num_adversaries']] or (
                            common.action_equal(good_actions[i][n], act_n[n + args['num_adversaries']], args['continuous_action'])):
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
            #print(opa_n)
            end = time.time()
            opa_time += end - start


            start = time.time()
            if args['reward_normalize']:
                normalize_reward = reward * 1.0 / args['done_reward']
            else:
                normalize_reward = reward

            buffer_s.append(observation)
            buffer_a.append(act_n)
            buffer_r.append(normalize_reward)
            buffer_o.append(opt_n)
            buffer_t.append(term_n)
            buffer_d.append(done)

            if (step != 0 and step % args['batch_size'] == 0) or step > args['epi_step'] or False not in done:
                buffer_s = np.array(buffer_s).swapaxes(0, 1)
                buffer_a = np.array(buffer_a).swapaxes(0, 1)
                buffer_r = np.array(buffer_r).swapaxes(0, 1)
                buffer_o = np.array(buffer_o).swapaxes(0, 1)
                buffer_t = np.array(buffer_t).swapaxes(0, 1)
                buffer_d = np.array(buffer_d).swapaxes(0, 1)
                for n in range(N_O):
                    index = 0
                    ppo_done_i = False
                    for d in range(len(buffer_d[n])):
                        index += 1
                        if buffer_d[n][d]:
                            if not ppo_done[n]:
                                ppo_done_i = True
                            break
                    if done[n]:
                        v_s_ = 0
                    else:
                        v_s_ = ppo_list[n].get_v(observation_[n])
                    discounted_r = []
                    for r in buffer_r[n][::-1]:
                        v_s_ = r + args['reward_decay'] * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    bs, ba, br, bo, bt = np.vstack(buffer_s[n][0: index]), buffer_a[n][0: index], np.array(
                        discounted_r)[:, np.newaxis][0: index], buffer_o[n][0: index], buffer_t[n][0: index]
                    if n < args['num_adversaries']:
                        actor = ppo_list[0: args['num_adversaries']]
                    else:
                        actor = ppo_list[args['num_adversaries']: N_O]
                    if not ppo_done[n]:
                        if n < args['num_adversaries'] and not args['adv_load_model']:
                            ppo_list[n].update(actor, bs, ba, br, bo, bt, episode, n)
                        elif n >= args['num_adversaries'] and not args['good_load_model']:
                            ppo_list[n].update(actor, bs, ba, br, bo, bt, episode, n)
                        ppo_done[n] = ppo_done_i
                buffer_s, buffer_a, buffer_r, buffer_o, buffer_t, buffer_d = [], [], [], [], [], []

            observation_ = np.array(observation_)
            option_obs_ = np.array(option_obs_)
            for n in range(N_O):
                if option_list[n] is not None and not done_n[n]:
                    # FIXME 1104 - store experiences separated by [agent_id]
                    if n < args['num_adversaries']:
                        option_list[n].store_transition(option_obs[n], act_n[n], normalize_reward[n], done[n],
                                                        option_obs_[n], opa_n[n], agent_id=n)
                    else:
                        option_list[n].store_transition(option_obs[n], act_n[n], normalize_reward[n], done[n],
                                                        option_obs_[n], opa_n[n], agent_id=n - args['num_adversaries'])
                if total_step > args['learning_step']:
                    if option_list[n] is not None and not done_n[n]:
                        if n < args['num_adversaries']:
                            actor = ppo_list[0: args['num_adversaries']]
                        else:
                            actor = ppo_list[args['num_adversaries']: N_O]
                        # TODO - update option functions
                        if n < args['num_adversaries']:
                            option_list[n].update(option_obs[n], opt_n[n], normalize_reward[n], done[n],
                                                  option_obs_[n], actor, agent_id=n)
                        else:
                            option_list[n].update(option_obs[n], opt_n[n], normalize_reward[n], done[n],
                                                  option_obs_[n], actor, agent_id=n - args['num_adversaries'])

            end = time.time()
            update_time += end - start

            # swap observation
            observation = observation_
            option_obs = option_obs_
            reward = np.array(reward)

            episode_discount_reward = episode_discount_reward + reward * np.power(args['reward_decay'], step)
            episode_reward = episode_reward + reward

            for n in range(N_O):
                if not done_n[n]:
                    done_n[n] = done[n]

            # break while loop when end of this episode
            if False not in done or step > args['epi_step']:
                discount_memory[episode % args['reward_memory']] = episode_discount_reward
                memory[episode % args['reward_memory']] = episode_reward
                if reward_memory_size < args['reward_memory']:
                    reward_memory_size += 1
                mean_memory = np.sum(memory, axis=0) / reward_memory_size
                discount_mean_memory = np.sum(discount_memory, axis=0) / reward_memory_size
                # print(RL.memory_counter)

                OJ.update([done_n, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory, episode])
                OJ.print_first()
                print("action_time: %.3f s" % action_time, "step_time: %.3f s" % step_time, "opa_time: %.3f s" % opa_time, "update_time: %.3f s" % update_time)

                logger.write_tb_log('discount_reward', episode_discount_reward, episode)
                logger.write_tb_log('discount_reward_mean', discount_mean_memory, episode)
                logger.write_tb_log('undiscounted reward', episode_reward, episode)
                logger.write_tb_log('reward_mean', mean_memory, episode)

                if total_step > args['learning_step']:
                    if adv_option is not None:
                        adv_option.update_e()
                    if good_option is not None:
                        good_option.update_e()

                break

            step += 1
            total_step += 1

        if args['save_model'] and episode % args['save_per_episodes'] == 0:
            OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])
            for n in range(N_O):
                ppo_list[n].save_model(args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(episode) + "_" + str(n))
            if adv_option is not None:
                adv_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/adv_option" + "_" + str(episode))
            if good_option is not None:
                good_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/good_option" + "_" + str(episode))

    if args['save_model']:
        for i in range(N_O):
            ppo_list[i].save_model(
                args['results_path'] + args['SAVE_PATH'] + "/model" + "_" + str(i))
        if adv_option is not None:
            adv_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/adv_option")
        if good_option is not None:
            good_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/good_option")

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

