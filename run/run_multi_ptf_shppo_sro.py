import numpy as np
from util.output_json import OutputJson
from alg.sharing_multi_ppo.ppo_add_entropy import PPO as PPO_add_entropy
from alg.sharing_multi_ppo.ppo import PPO
from alg.sharing_multi_ppo.option_ma_sro import SROption, SREmbedding
from alg.common.common import build_source_actor
import tensorflow as tf
from alg.common import common
import os


def run(args, env, RL, logger):
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

    adv_ppo = None
    good_ppo = None
    if args['num_adversaries'] > 0:
        n_actions, n_features = env.action_space[0].n, env.observation_space[0].shape[0]
        if args['adv_use_option']:
            adv_ppo = PPO_add_entropy(n_actions, n_features, N_O, args, SESS, logger)
        else:
            if not args['adv_load_model']:
                adv_ppo = PPO(n_actions, n_features, N_O, args, SESS, logger)
    if N_O - args['num_adversaries'] > 0:
        n_actions, n_features = env.action_space[args['num_adversaries']].n, env.observation_space[args['num_adversaries']].shape[0]
        if args['good_use_option']:
            good_ppo = PPO_add_entropy(n_actions, n_features, N_O, args, SESS, logger)
        else:
            if not args['good_load_model']:
                good_ppo = PPO(n_actions, n_features, N_O, args, SESS, logger)


    ppo_list = []
    for n in range(N_O):
        n_actions, n_features = env.action_space[n].n, env.observation_space[n].shape[0]
        # print(n_actions, n_features)
        if n < args['num_adversaries']:
            if args['adv_policy'] == "ppo" and args['adv_use_option']:
                ppo_list.append(adv_ppo)
            elif args['adv_policy'] == "ppo" and not args['adv_use_option']:
                if args['adv_load_model']:  # 参数和model一致
                    load_args = {}
                    load_args['policy'] = args['adv_policy']
                    load_args['action_dim'] = n_actions
                    load_args['features'] = n_features
                    ppo = build_source_actor(load_args, SESS, args['adv_load_model_path'], n)
                    ppo_list.append(ppo)
                else:
                    ppo_list.append(adv_ppo)
        else:
            if args['good_policy'] == "ppo" and args['good_use_option']:
                ppo_list.append(good_ppo)
            elif args['good_policy'] == "ppo" and not args['good_use_option']:
                if args['good_load_model']:  # 参数和model一致
                    load_args = {}
                    load_args['policy'] = args['good_policy']
                    load_args['action_dim'] = n_actions
                    load_args['features'] = n_features
                    ppo = build_source_actor(load_args, SESS, args['good_load_model_path'], n)
                    ppo_list.append(ppo)
                else:
                    ppo_list.append(good_ppo)

    SESS.run(tf.global_variables_initializer())

    for n in range(N_O):
        if n < args['num_adversaries']:
            if args['adv_load_model']:
                ppo_list[n].load_model(args['adv_load_model_path'] + '_' + str(n))
        else:
            if args['good_load_model']:
                ppo_list[n].load_model(args['good_load_model_path'] + '_' + str(n))

    total_step = 0
    memory = np.zeros((args['reward_memory'], env.n))
    reward_memory_size = 0
    discount_memory = np.zeros((args['reward_memory'], env.n))
    numGames = args['numGames']
    field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean','episode']
    OJ = OutputJson(field)
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
        while True:
            act_n = []
            game_acts = []

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
                action = ppo_list[n].choose_action(observation[n], n)
                act_n.append(action)
                game_acts.append(action)

            # FIXME 1106 - to check
            # observation_, reward, done, _ = env.step(game_acts)
            observation_, reward, done, _ = env.step(np.copy(game_acts))

            option_obs_ = observation_

            #print(done)
            opa_n = [[] for i in range(N_O)]
            if args['adv_use_option']:
                if args['other_option_update']:
                    adv_actions = []
                    adv_obs = observation[0: args['num_adversaries']]
                    adv_obs = [list(temp) for temp in adv_obs]
                    for n in range(args['num_adversaries']):
                        adv_actions.append(ppo_list[n].choose_deterministic_action(adv_obs, n))
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
                        good_actions.append(ppo_list[n].choose_deterministic_action(good_obs, n))
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
                buffer_discounted_r = []

                for n in range(N_O):
                    index = len(buffer_d[n])
                    if done[n]:
                        v_s_ = 0
                    else:
                        v_s_ = ppo_list[n].get_v(observation_[n], n)
                    discounted_r = []
                    for r in buffer_r[n][::-1]:
                        v_s_ = r + args['reward_decay'] * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    buffer_discounted_r.append(discounted_r)
                if adv_ppo is not None:
                    actor = range(args['num_adversaries'])
                    bs = []#np.reshape(buffer_s[0: args['num_adversaries']], (-1, len(buffer_s[0][0])))
                    ba = []#np.reshape(buffer_a[0: args['num_adversaries']], (-1, len(buffer_a[0][0])))
                    br = []#np.reshape(buffer_discounted_r[0: args['num_adversaries']], (-1, 1))
                    bo = []#np.reshape(buffer_o[0: args['num_adversaries']], (-1, len(buffer_o[0][0])))
                    bt = []#np.reshape(buffer_t[0: args['num_adversaries']], (-1, len(buffer_t[0][0])))
                    agent_list = []
                    for i in range(args['num_adversaries']):
                        for j in range(len(buffer_s[i])):
                            agent_list.append(i)
                            bs.append(buffer_s[i][j])
                            ba.append(buffer_a[i][j])
                            br.append([buffer_discounted_r[i][j]])
                            bo.append(buffer_o[i][j])
                            bt.append(buffer_t[i][j])
                    adv_ppo.update(actor, bs, ba, br, bo, bt, episode, agent_list)
                if good_ppo is not None:
                    actor = range(args['num_adversaries'], N_O)
                    bs = []#np.reshape(buffer_s[args['num_adversaries']: N_O], (-1, len(buffer_s[args['num_adversaries']][0])))
                    ba = []#np.reshape(buffer_a[args['num_adversaries']: N_O], (-1, len(buffer_a[args['num_adversaries']][0])))
                    br = []#np.reshape(buffer_discounted_r[args['num_adversaries']: N_O], (-1, 1))
                    bo = []#np.reshape(buffer_o[args['num_adversaries']: N_O], (-1, len(buffer_o[args['num_adversaries']][0])))
                    bt = []#np.reshape(buffer_t[args['num_adversaries']: N_O], (-1, len(buffer_t[args['num_adversaries']][0])))
                    agent_list = []
                    for i in range(args['num_adversaries'], N_O):
                        for j in range(len(buffer_s[i])):
                            agent_list.append(i)
                            bs.append(buffer_s[i][j])
                            ba.append(buffer_a[i][j])
                            br.append([buffer_discounted_r[i][j]])
                            bo.append(buffer_o[i][j])
                            bt.append(buffer_t[i][j])
                    good_ppo.update(actor, bs, ba, br, bo, bt, episode, agent_list)
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

            # swap observation
            observation = observation_
            option_obs = option_obs_
            reward = np.array(reward)

            episode_discount_reward = episode_discount_reward + reward * np.power(args['reward_decay'], step)
            episode_reward = episode_reward + reward

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
            if adv_ppo is not None:
                adv_ppo.save_model(args['results_path'] + args['SAVE_PATH'] + "/adv_model" + "_" + str(episode))
            if good_ppo is not None:
                good_ppo.save_model(args['results_path'] + args['SAVE_PATH'] + "/good_model" + "_" + str(episode))
            if adv_option is not None:
                adv_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/adv_option" + "_" + str(episode))
            if good_option is not None:
                good_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/good_option" + "_" + str(episode))

    if args['save_model']:
        if adv_ppo is not None:
            adv_ppo.save_model(args['results_path'] + args['SAVE_PATH'] + "/adv_model")
        if good_ppo is not None:
            good_ppo.save_model(args['results_path'] + args['SAVE_PATH'] + "/good_model")
        if adv_option is not None:
            adv_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/adv_option")
        if good_option is not None:
            good_option.save_model(args['results_path'] + args['SAVE_PATH'] + "/good_option")

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])

