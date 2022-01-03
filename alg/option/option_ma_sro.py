import tensorflow as tf
import numpy as np
from alg.optimizer import Optimizer
from util.ReplayBuffer import ShareReplayBuffer as ReplayBuffer


class SREmbedding:
    def __init__(self, option_dim, agent_num, n_features, args, sess, logger, name,
                 emb_dim=32):
        self.name = name
        self.args = args
        self.agent_num = agent_num
        self.option_dim = option_dim
        self.n_features = n_features
        self.logger = logger

        self.emb_dim = emb_dim

        self.update_step = 0
        opt = Optimizer(args['optimizer'], args['learning_rate_r'])
        self.Opt = opt.get_optimizer()

        with tf.variable_scope(self.name + '_train_input'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
            self.reward = tf.placeholder(tf.float32, [None], name='r')
            self.agent_id = tf.placeholder(tf.int32, [None], name='aid')

            self.sr = tf.placeholder(tf.float32, [None, self.emb_dim], name='sr')

        self.w_init = tf.random_normal_initializer(0., .01)

        # FIXME 1104 - build embedding networks
        self.emb = self._build_emb(self.name + '_embedding', self.s)
        self.recon = self._build_recon(self.name + '_reconstruction', self.emb)
        self.imed_r = self._build_reward(self.name + '_reward', self.emb)

        self.q_sr_raw = self._build_reward(self.name + '_reward', self.sr, reuse=True)
        self.q_sr_indiv = tf.reduce_sum(self.q_sr_raw * tf.one_hot(self.agent_id, self.agent_num), axis=-1, keepdims=True)

        with tf.variable_scope(self.name + '_embedding_loss'):
            self.recon_loss = tf.reduce_mean(tf.square(self.s - self.recon))
            corresponding_imed_r = tf.reduce_sum(tf.one_hot(self.agent_id, self.agent_num) * self.imed_r, axis=-1)
            self.r_loss = tf.reduce_mean(tf.square(self.reward - corresponding_imed_r))
            # FIXME 1104 - tune the coef if necessary
            self.r_loss_total = args['recon_loss_coef'] * self.recon_loss + self.r_loss

        with tf.name_scope(self.name + '_grad'):
            # FIXME 1104 - gradients for embedding and reward weights
            gradients_emb_r = self.Opt.compute_gradients(self.r_loss_total)
            for i, (grad, var) in enumerate(gradients_emb_r):
                if grad is not None:
                    gradients_emb_r[i] = (tf.clip_by_norm(grad, args['clip_value']), var)
            self.update_emb_r = self.Opt.apply_gradients(gradients_emb_r)

        self.sess = sess

    def get_embedding(self, s):
        s = np.array(s).reshape(-1, self.n_features)
        if s.shape[0] == 1:
            emb = self.sess.run(self.emb, feed_dict={self.s: s})[0]
        else:
            emb = self.sess.run(self.emb, feed_dict={self.s: s})
        return emb

    # size: sr [-1, emb_dim]
    def get_Q_sr(self, sr, agent_id):
        aid = np.tile(np.array(agent_id), sr.shape[0])
        q_sr_raw, q_sr_indiv = self.sess.run([self.q_sr_raw, self.q_sr_indiv],
                                             feed_dict={self.sr: sr,
                                                        self.agent_id: aid,
                                                        })
        return q_sr_raw, q_sr_indiv

    def update(self, s, r, agent_id):
        # FIXME 1104 - update embedding and reward weights with (s,r,agent_id)
        loss_emb_r, _ = self.sess.run([self.r_loss_total, self.update_emb_r],
                                      feed_dict={
                                          self.s: s,
                                          self.reward: r,
                                          self.agent_id: agent_id,
                                      })

        self.logger.write_tb_log(self.name + '_emb_loss', loss_emb_r, self.update_step)
        self.update_step += 1


    # FIXME 1104 - define embedding networks and reward weights
    def _build_emb(self, scope, s, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope(scope, reuse=reuse):
            l1 = tf.layers.dense(s, self.args['option_embedding_layer'], activation=tf.nn.relu, name='l1',
                                  trainable=trainable, kernel_initializer=self.w_init)
            # l2 = tf.layers.dense(l1, self.args['option_embedding_layer'], activation=tf.nn.relu, name='l2',
            #                      trainable=trainable, kernel_initializer=self.w_init)
            m = tf.layers.dense(l1, self.emb_dim, activation=None, name='m', trainable=trainable,
                                kernel_initializer=self.w_init)
            return m

    def _build_recon(self, scope, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope(scope, reuse=reuse):
            l1 = tf.layers.dense(m, self.args['option_embedding_layer'], activation=tf.nn.relu, name='l1',
                                  trainable=trainable, kernel_initializer=self.w_init)
            # l2 = tf.layers.dense(l1, self.args['option_embedding_layer'], activation=tf.nn.relu, name='l2',
            #                      trainable=trainable, kernel_initializer=self.w_init)
            recon = tf.layers.dense(l1, self.n_features, activation=None, name='recon', trainable=trainable,
                                    kernel_initializer=self.w_init)
            return recon

    def _build_reward(self, scope, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope(scope, reuse=reuse):
            r = tf.layers.dense(m, self.agent_num, activation=None, name='reward', trainable=trainable, use_bias=False,
                                kernel_initializer=self.w_init)
            return r
            # r_list = []
            # for i in range(self.agent_num):
            #     r = tf.layers.dense(m, 1, activation=None, name='reward_' + str(i), trainable=trainable, use_bias=False,
            #                         kernel_initializer=self.w_init)
            #     r_list.append(r)
            # return r_list


class SROption:
    def __init__(self, option_dim, agent_num, n_features, args, sess, logger, name,
                 sr_emb=None):
        self.name = name
        self.args = args
        self.agent_num = agent_num
        self.option_dim = option_dim
        self.n_features = n_features
        self.logger = logger

        self.sr_emb = sr_emb
        self.emb_dim = sr_emb.emb_dim

        self.update_step = 0
        self.replace_target_iter = args['replace_target_iter']
        self.e_greedy = args['e_greedy']
        self.epsilon_increment = args['e_greedy_increment']
        self.epsilon = args['start_greedy'] if args['e_greedy_increment'] != 0 else self.e_greedy

        # FIXME 1104
        opt0 = Optimizer(args['optimizer'], args['learning_rate_o'])
        self.Opt_O = opt0.get_optimizer()
        opt1 = Optimizer(args['optimizer'], args['learning_rate_t'])
        self.Opt_T = opt1.get_optimizer()
        # opt0 = [Optimizer(args['optimizer'], args['learning_rate_o']) for _ in range(self.agent_num)]
        # self.Opt_O = [opt0[i].get_optimizer() for i in range(self.agent_num)]
        # opt1 = [Optimizer(args['optimizer'], args['learning_rate_t']) for _ in range(self.agent_num)]
        # self.Opt_T = [opt1[i].get_optimizer() for i in range(self.agent_num)]

        # FIXME 1105 - set buffers for each agent
        self.replay_buffer = ReplayBuffer(args['memory_size'])
        # self.replay_buffer = ReplayBufferSR(args['memory_size'])

        with tf.variable_scope(self.name + '_train_input'):
            # self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
            # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
            self.option_o = tf.placeholder(tf.int32, [None], name='op')
            self.option_a_t = tf.placeholder(tf.float32, [None, self.option_dim], name='op_at')
            # self.reward = tf.placeholder(tf.float32, [None], name='r')
            self.done = tf.placeholder(tf.float32, [None], name='done')

            # FIXME 1105 - new placehodler
            self.emb = tf.placeholder(tf.float32, [None, self.emb_dim], name='embedding_for_s')
            self.emb_ = tf.placeholder(tf.float32, [None, self.emb_dim], name='embedding_for_s_')
            self.agent_id = tf.placeholder(tf.int32, [None], name='agent_id')
            self.option_o_max = tf.placeholder(tf.int32, [None], name='op_o_max')
            self.advantage = [tf.placeholder(tf.float32, [None], name='advantage_for_term_update')
                              for i in range(self.agent_num)]

        self.w_init = tf.random_normal_initializer(0., .01)

        # FIXME 1104 - build sr networks
        # FIXME 1104 - reshaping for following calculation
        #   [-1, option_dim * emb_dim] ---> [-1, option_dim, emb_dim], 2D ---> 3D
        self.sr_eval_list, self.term_eval_list, self.sr_eval_reshape_list, self.term_eval_reshape_list\
            = self._build_sr(self.name + '_sr_w_term_net', self.emb)
        self.sr_target_list, self.term_target_list, self.sr_target_reshape_list, self.term_target_reshape_list\
            = self._build_sr(self.name + '_sr_w_term_target', self.emb_)
        self.sr_next_eval_list, self.term_next_eval_list, self.sr_next_eval_reshape_list, self.term_next_eval_reshape_list\
            = self._build_sr(self.name + '_sr_w_term_net', self.emb_, reuse=True)

        # self.q_omega_current, self.term_current = self._build_net(self.name + '_q_net', self.s)
        # self.q_omega_target, self.term_target = self._build_net(self.name + '_q_target', self.s_)
        # self.q_omega_next_current, self.term_next_current = self._build_net(self.name + '_q_net', self.s_, reuse=True)

        # FIXME 1104 - collect parameters
        self.sr_term_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_sr_w_term_net')
        self.target_sr_term_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_sr_w_term_target')

        # self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_q_net')
        # self.target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '_q_target')

        # TODO - update rules for Q(s,o):
        #   Q(s,o) = r + \gamma * U(s',o')
        #   U(s,o) = b(s,o) * Q_max(s,o) + (1 - b(s,o) * Q(s,o))
        #   ---> SR
        #   m(\phi(s),o) = \phi(s) + \gamma * U(\phi(s)',o')
        #   U(\phi(s),o) = b(\phi(s),o) * m(\phi(s),o_max) + (1 - b(\phi(s),o) * m(\phi(s),o))
        #       o_max = argmax_{o} m(\phi(s),o) * w
        with tf.variable_scope(self.name + '_sr_value'):
            # size = [-1, 1]
            term_val_next_list = [tf.reduce_sum(self.term_next_eval_list[i] * tf.one_hot(self.option_o,
                                                                                         self.option_dim),
                                                axis=-1, keepdims=True)
                                  for i in range(self.agent_num)]
            # size = [-1, emb_dim]
            # self.sr_val_next_list = [tf.reduce_sum(self.sr_next_eval_reshape_list[i]
            #                                   * tf.reshape(tf.one_hot(self.option_o, self.option_dim),
            #                                                (-1, self.option_dim, 1)), axis=1)
            #                     for i in range(self.agent_num)]

            # FIXME 1104 - calculate: m1(\phi(s'),o_max), with o_max = argmax_{o} m1(\phi(s'),o) * w1
            #   size: [-1, 1, emb_dim]
            max_o_sr_next_targ = [tf.reduce_sum(self.sr_target_reshape_list[i]
                                                * tf.reshape(tf.one_hot(self.option_o_max, self.option_dim),
                                                             (-1, self.option_dim, 1)), axis=1, keepdims=True)
                                  for i in range(self.agent_num)]

            # term_val_next = tf.reduce_sum(self.term_next_current * tf.one_hot(self.option_o, self.option_dim), axis=-1)
            # q_omega_val_next = tf.reduce_sum(self.q_omega_next_current * tf.one_hot(self.option_o, self.option_dim), axis=-1)
            # max_q_omega_next = tf.reduce_max(self.q_omega_next_current, axis=-1)
            # max_q_omega_next_targ = tf.reduce_sum(
            #     self.q_omega_target * tf.one_hot(tf.argmax(self.q_omega_next_current, axis=-1), self.option_dim), axis=-1)

        with tf.variable_scope(self.name + '_sr_loss'):
            # FIXME 1104 -
            #   size: [-1, option_dim, emb_dim]
            u_next_raw_list = [(1 - self.term_next_eval_reshape_list[i]) * self.sr_target_reshape_list[i]
                               + self.term_next_eval_reshape_list[i] * max_o_sr_next_targ[i]
                               for i in range(self.agent_num)]
            u_next_list = [tf.stop_gradient(u_next_raw_list[i] * tf.reshape(1 - self.done, (-1, 1, 1)))
                           for i in range(self.agent_num)]
            # FIXME 1104 - to check
            self.sr_loss = [tf.reduce_mean(
                tf.reduce_sum(tf.reshape(self.option_a_t, (-1, self.option_dim, 1))
                              * tf.losses.mean_squared_error(tf.reshape(self.emb, (-1, 1, self.emb_dim))
                                                             + self.args['reward_decay'] * u_next_list[i],
                                                             self.sr_eval_reshape_list[i],
                                                             reduction=tf.losses.Reduction.NONE),
                              axis=-1),
                axis=-1) for i in range(self.agent_num)]

        # TODO - update rules for termination:
        #   todo ...
        with tf.variable_scope(self.name + '_term_loss'):
            # TODO 1105 - because cannot calculate Q values directly, we do this part in update() for clarity
            # if self.args['xi'] == 0:
            #     # FIXME 1104 - if top k, we needs extra calculation (TODO)
            #     # xi = 0.8 * (max_q_omega_next - tf.nn.top_k(self.q_omega_next_current, 2)[0][:, 1])
            #     xi = self.args['xi']
            # else:
            #     xi = self.args['xi']
            # advantage_go = q_omega_val_next - max_q_omega_next + xi
            # advantage = tf.stop_gradient(advantage_go)
            # self.total_error_term = term_val_next * advantage

            self.total_error_term = [term_val_next_list[i] * self.advantage[i][..., None]
                                     for i in range(self.agent_num)]

        with tf.name_scope(self.name + '_grad'):
            self.update_o_list, self.update_t_list = [], []
            for i in range(self.agent_num):
                gradients = self.Opt_O.compute_gradients(self.sr_loss[i], var_list=self.sr_term_vars)
                for k, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[k] = (tf.clip_by_norm(grad, args['option_clip_value']), var)
                update_o = self.Opt_O.apply_gradients(gradients)
                gradients_t = self.Opt_T.compute_gradients(self.total_error_term[i], var_list=self.sr_term_vars)
                for k, (grad, var) in enumerate(gradients_t):
                    if grad is not None:
                        gradients_t[k] = (tf.clip_by_norm(grad, args['option_clip_value']), var)
                update_t = self.Opt_T.apply_gradients(gradients_t)

                self.update_o_list.append(update_o)
                self.update_t_list.append(update_t)

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.target_sr_term_vars, self.sr_term_vars)]

        self.sess = sess

    # FIXME 1104 - define sr networks and term networks for agents
    def _build_sr(self, scope, emb, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope(scope, reuse=reuse):
            l_a = tf.layers.dense(emb, self.args['option_layer_1'], tf.nn.relu6, kernel_initializer=self.w_init,
                                  name='la', trainable=trainable)
            l_a_2 = tf.layers.dense(l_a, self.args['option_layer_2'], tf.nn.relu6, kernel_initializer=self.w_init,
                                    name='la_2', trainable=trainable)

            with tf.variable_scope("option_sr"):
                # FIXME - why tanh activation ?
                sr_list, sr_resh_list = [], []
                for i in range(self.agent_num):
                    sr_omega = tf.layers.dense(l_a_2, self.option_dim * self.emb_dim, None, trainable=trainable,
                                               kernel_initializer=self.w_init, name='omega_sr_' + str(i))
                    sr_list.append(sr_omega)
                    sr_resh_list.append(tf.reshape(sr_omega, (-1, self.option_dim, self.emb_dim)))

            with tf.variable_scope("termination_prob"):
                term_list, term_resh_list = [], []
                for i in range(self.agent_num):
                    term_prob = tf.layers.dense(l_a_2, self.option_dim, tf.sigmoid, trainable=trainable,
                                                kernel_initializer=self.w_init, name='term_prob_' + str(i))
                    term_list.append(term_prob)
                    term_resh_list.append(tf.reshape(term_prob, (-1, self.option_dim, 1)))
            return sr_list, term_list, sr_resh_list, term_resh_list


    def store_transition(self, observation, action, reward, done, observation_, opa, agent_id):
        self.replay_buffer.add(observation, action, reward, done, observation_, opa, agent_id)

    def update_e(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.e_greedy else self.e_greedy

    def choose_o(self, s, agent_id):
        if np.random.uniform() < self.epsilon:
            emb = self.sr_emb.get_embedding(s)
            sr = self.sess.run(self.sr_eval_list[agent_id], feed_dict={self.emb: emb[np.newaxis, :]})
            _, q = self.sr_emb.get_Q_sr(np.reshape(sr, (-1, self.emb_dim)), agent_id)

            return np.argmax(q)
            # options = self.sess.run(self.q_omega_current, feed_dict={self.s: s[np.newaxis, :]})
            # options = options[0]
            # return np.argmax(options)
        else:
            return np.random.randint(0, self.option_dim)

    def get_t(self, s_, option, agent_id):
        emb_ = self.sr_emb.get_embedding(s_)
        terminations = self.sess.run(self.term_next_eval_list[agent_id],
                                     feed_dict={self.emb_: emb_[np.newaxis, :]})
        return terminations[0][option]

    def get_term_prob(self, s, agent_id):
        emb = self.sr_emb.get_embedding(s)
        return self.sess.run(self.term_eval_list[agent_id], feed_dict={self.emb: emb})

    # FIXME 1105 - to check
    def update(self, observation, option, reward, done, observation_, actor, agent_id):
        if self.update_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        self.update_step += 1
        # FIXME 1105 - update termination function, i.e., b(\phi(s), o)
        if not done:
            # emb = self.sr_emb.get_embedding(observation)
            emb_ = self.sr_emb.get_embedding(observation_)
            # size = [1, option_dim * emb_dim]
            sr_next = self.sess.run(self.sr_next_eval_list[agent_id], feed_dict={self.emb_: emb_[np.newaxis, :]})
            # size = [option_dim, emb_dim] ---> [option_dim, 1]
            _, q_sr_next = self.sr_emb.get_Q_sr(np.reshape(sr_next, (-1, self.emb_dim)), agent_id)
            # q_sr_next = q_sr_next.swapaxes(0, 1)
            q_sr_val_next = q_sr_next[option]
            max_q_sr_next = np.max(q_sr_next, axis=0)
            # TODO - add xi
            advantage = q_sr_val_next - max_q_sr_next

            loss_term, _ = self.sess.run([self.total_error_term[agent_id], self.update_t_list[agent_id]],
                                         feed_dict={self.emb_: emb_[np.newaxis, :],
                                                    self.option_o: [option],
                                                    self.advantage[agent_id]: advantage + self.args['xi'],
                                                    # self.done: [1.0 if done == True else 0.0]
                                                    })

            self.logger.write_tb_log(self.name + '_t_loss' + str(agent_id), loss_term, self.update_step)
        # FIXME 1105 - update SR function, i.e., m(\phi(s), o)
        minibatch = self.replay_buffer.get_batch(self.args['option_batch_size'])
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        done_batch = np.array([1.0 if data[3] else 0.0 for data in minibatch], dtype=np.float32)
        next_state_batch = np.asarray([data[4] for data in minibatch])
        opa_batch = np.asarray([data[5] for data in minibatch])
        agent_id_batch = np.asarray([data[6] for data in minibatch])
        '''
        opa_batch = []
        actor_action_batch = []
        for i in range(self.option_dim):
            actor_action_batch.append(actor[i].choose_deterministic_action(state_batch))
        for act in range(self.args['option_batch_size']):
            opa = []
            for i in range(self.option_dim):
                if not self.args['continuous_action']:
                    if common.action_equal(actor_action_batch[i][act], action_batch[act],
                                           self.args['continuous_action']):
                        opa.append(1)
                    else:
                        opa.append(0)
                else:
                    if common.action_equal([actor_action_batch[i][0][act], actor_action_batch[i][1][act]],
                                           action_batch[act], self.args['continuous_action']):
                        opa.append(1)
                    else:
                        opa.append(0)
            opa_batch.append(opa)
        '''
        # opa_batch = np.asarray([data[5] for data in minibatch])
        # aid_batch = np.asarray([data[6] for data in minibatch])
        emb_batch = self.sr_emb.get_embedding(state_batch)
        next_emb_batch = self.sr_emb.get_embedding(next_state_batch)
        # size = [-1, option_dim, emb_dim]
        sr_next_eval = self.sess.run(self.sr_next_eval_reshape_list[agent_id], feed_dict={self.emb_: next_emb_batch})
        # size = [-1 * option_dim, emb_dim] ---> [-1 * option_dim, 1]
        _, q_sr_next_eval = self.sr_emb.get_Q_sr(np.reshape(sr_next_eval, (-1, self.emb_dim)), agent_id)
        # size = [-1 * option_dim, 1] ---> [-1, option_dim] ---> [-1,]
        op_omax_batch = np.argmax(np.reshape(q_sr_next_eval, (-1, self.option_dim)), axis=1)

        loss_q_omega, _ = self.sess.run([self.sr_loss[agent_id], self.update_o_list[agent_id]],
                                        feed_dict={
                                            self.emb: emb_batch,
                                            self.emb_: next_emb_batch,
                                            # self.option_o: [option],
                                            self.done: done_batch,
                                            self.option_a_t: opa_batch,
                                            self.option_o_max: op_omax_batch,
                                        })

        # FIXME 1104 - update embedding and reward weights with (s,r,agent_id)
        self.sr_emb.update(state_batch, reward_batch, agent_id_batch)
        # loss_emb_r, _ = self.sess.run([self.r_loss_total, self.update_emb_r], feed_dict={
        #     self.s: state_batch,
        #     self.reward: reward_batch,
        #     self.agent_id: aid_batch
        # })

        self.logger.write_tb_log(self.name + '_o_loss' + str(agent_id), loss_q_omega, self.update_step)

    def load_model(self, path):
        saver = tf.train.Saver(self.sr_term_vars)
        saver.restore(self.sess, path + ".ckpt")

    def save_model(self, path):
        saver = tf.train.Saver(self.sr_term_vars)
        saver.save(self.sess, path + ".ckpt")
