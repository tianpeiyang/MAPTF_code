import numpy as np
import tensorflow as tf
from alg.optimizer import Optimizer


class PPO:
    def __init__(self, n_actions, n_features, n_agents, args, SESS, logger):
        self.args = args
        self.n_actions = n_actions
        self.n_features = n_features + n_agents
        self.n_agents = n_agents
        self.logger = logger
        self.learning_step = 0

        self.obs = tf.placeholder(tf.float32, [None, self.n_features], 's')

        self.act_probs, self.policy_param = self.build_actor_net(self.args['policy'])
        self.o_act_probs, self.o_policy_param = self.build_actor_net(self.args['old_policy'])
        self.v_preds, self.v_param = self.build_critic_net('critic')

        if self.args['continuous_action']:
            self.sample_action = tf.squeeze(self.act_probs.sample(1), axis=0)
        else:
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

        self.replace_op = [tf.assign(t, e) for t, e in zip(self.o_policy_param, self.policy_param)]

        opt = Optimizer(args['optimizer'], args['learning_rate_a'])
        self.optimizer = opt.get_optimizer()
        opt_c = Optimizer(args['optimizer'], args['learning_rate_c'])
        self.optimizer_c = opt_c.get_optimizer()

        with tf.variable_scope('train_inp'):
            if self.args['continuous_action']:
                self.actions = tf.placeholder(tf.float32, [None, self.n_actions], 'action')
                self.mu = tf.placeholder(tf.float32, [None, self.n_actions], 'input_mu')
                self.sigma = tf.placeholder(tf.float32, [None, self.n_actions], 'input_sigma')
            else:
                self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='actions')
                self.s_a_prob = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='s_a_prob')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='gaes')
            self.term = tf.placeholder(dtype=tf.float32, shape=[None], name='term')
            self.e = tf.placeholder(tf.float32, (), 'e')

        self.build_loss()

        self.sess = SESS

    def build_actor_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(inputs=self.obs, units=self.args['n_layer_a_1'], activation=tf.nn.relu, trainable=trainable)
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.args['n_layer_a_2'], activation=tf.nn.relu,
                                      trainable=trainable)
            if self.args['continuous_action']:
                mu = self.args['action_clip'] * tf.layers.dense(inputs=layer_2, units=self.n_actions, activation=tf.nn.tanh, trainable=trainable)
                sigma = tf.layers.dense(inputs=layer_2, units=self.n_actions, activation=tf.nn.softplus, trainable=trainable)
                act_probs = tf.distributions.Normal(loc=mu, scale=sigma + 1e-9)
            else:
                act_probs = tf.layers.dense(inputs=layer_2, units=self.n_actions, activation=tf.nn.softmax)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return act_probs, params

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            layer_1 = tf.layers.dense(inputs=self.obs, units=self.args['n_layer_c_1'], activation=tf.nn.relu)
            layer_2 = tf.layers.dense(inputs=layer_1, units=self.args['n_layer_c_2'], activation=tf.nn.relu)
            v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return v_preds, params

    def build_loss(self):
        with tf.variable_scope('update_critic'):
            self.advantage = self.rewards - self.v_preds
            self.c_loss = tf.reduce_mean(tf.square(self.advantage))
            self.train_c_op = self.optimizer_c.minimize(self.c_loss, var_list=self.v_param)

        with tf.variable_scope('update_actor'):
            if self.args['continuous_action']:
                act_probs = self.act_probs.prob(self.actions)
                act_probs_old = self.o_act_probs.prob(self.actions)
                entropy = self.act_probs.entropy()
                otherNormal = tf.distributions.Normal(self.mu, self.sigma)
                otherEntroy = otherNormal.cross_entropy(self.act_probs)
            else:
                act_probs = self.act_probs * self.actions #tf.one_hot(indices=self.actions, depth=self.act_probs.shape[1])
                act_probs = tf.reduce_sum(act_probs, axis=1)
                # probabilities of actions which agent took with old policy
                act_probs_old = self.o_act_probs * self.actions #tf.one_hot(indices=self.actions, depth=self.o_act_probs.shape[1])
                act_probs_old = tf.reduce_sum(act_probs_old, axis=1)
                entropy = -tf.reduce_sum(self.act_probs *
                                         tf.log(tf.clip_by_value(self.act_probs, 1e-9, 1.0)), axis=1)
                #otherEntroy = -self.s_a_prob * tf.log(self.act_probs + 1e-9)
                otherEntroy = -self.s_a_prob * tf.log(tf.clip_by_value(self.act_probs, 1e-9, 1.0))

            with tf.variable_scope('loss/clip'):
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.args['clip_value'],
                                                  clip_value_max=1 + self.args['clip_value'])
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                self.loss_clip = tf.reduce_mean(loss_clip)

                self.entropy = self.args['c2'] * tf.reduce_mean(entropy)  # mean of entropy of pi(obs)

                t = tf.reshape(self.term, shape=[-1, 1])
                entropyTS = tf.reduce_sum(otherEntroy, axis=1,
                                           keepdims=True)
                weight = 0.5 + tf.tanh(3 - self.args['c3'] * self.e) / 2
                entropyTS = entropyTS * weight * self.args['c1']
                self.entropyTS = tf.reduce_mean(entropyTS)

                self.a_loss = -(self.loss_clip + self.entropy) + self.entropyTS
                self.train_a_op = self.optimizer.minimize(self.a_loss, var_list=self.policy_param)

    def get_agent_obs(self, obs, agent_id=0):
        if type(agent_id) is int:
            agent_id_arr = [agent_id] * len(obs)
        elif type(agent_id) is list:
            agent_id_arr = agent_id
        else:
            raise Exception('the agent_id field must be type of int or list')
        agent_one_hot = np.eye(self.n_agents)[agent_id_arr]
        obs = np.hstack((agent_one_hot, obs))
        return obs

    def choose_action(self, obs, agent_id=0):
        obs = np.array(obs)
        obs = obs[np.newaxis, :]
        obs = self.get_agent_obs(obs, agent_id)
        if self.args['continuous_action']:
            actions, v_preds = self.sess.run([self.sample_action, self.v_preds], {self.obs: obs})
            return np.clip(actions[0], -self.args['action_clip'], self.args['action_clip'])
        else:
            if self.args['stochastic']:
                actions, v_preds, p = self.sess.run([self.act_stochastic, self.v_preds, self.act_probs], feed_dict={self.obs: obs})
                action = actions[0]
                action_one_hot = np.zeros(self.n_actions)
                action_one_hot[action] = 1
                return action_one_hot
            else:
                actions, v_preds = self.sess.run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})
                action = actions[0]
                action_one_hot = np.zeros(self.n_actions)
                action_one_hot[action] = 1
                return action_one_hot

    def choose_hold_action(self, obs):
        return np.zeros(self.n_actions)

    def choose_deterministic_action(self, obs, agent_id=0):
        obs = self.get_agent_obs(obs, agent_id)
        actions = self.sess.run([self.act_deterministic], feed_dict={self.obs: obs})[0]
        action_one_hots = []
        for i in range(len(actions)):
            action = actions[i]
            action_one_hot = np.zeros(self.n_actions)
            action_one_hot[action] = 1
            action_one_hots.append(action_one_hot)
        return action_one_hots

    def choose_acton_prob(self, observation, agent_id=0):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        observation = self.get_agent_obs(observation, agent_id)
        if self.args['continuous_action']:
            actions_value = self.sess.run(self.act_probs, feed_dict={self.obs: observation})
            actions_value = [actions_value[0][0], actions_value[1][0]]
        else:
            actions_value = self.sess.run(self.act_probs, feed_dict={self.obs: observation})[0]
        return actions_value

    def get_v(self, s, agent_id=0):
        obs = np.array(s)
        obs = obs[np.newaxis, :]
        obs = self.get_agent_obs(obs, agent_id)
        return self.sess.run(self.v_preds, {self.obs: obs})[0, 0]

    def update(self, actor, s, a, r, options, terms, epi, agentid):
        self.sess.run(self.replace_op)

        source_actor_prob = []
        mu = []
        sigma = []
        for i, o in enumerate(options):
            o = actor[o]
            if o == agentid[i]:
                terms[i] = 0
            if self.args['continuous_action']:
                a_prob = self.choose_acton_prob(s[i], o)
                mu.append(a_prob[0])
                sigma.append(a_prob[1])
            else:
                if o == agentid[i]:
                    a_prob = self.choose_hold_action(s[i])
                else:
                    a_prob = self.choose_acton_prob(s[i], o)
                source_actor_prob.append(a_prob)
        s = self.get_agent_obs(s, agentid)
        adv = self.sess.run(self.advantage, {self.obs: s, self.rewards: r})
        if self.args['continuous_action']:
            for i in range(self.args['epi_train_times']):
                _, a_loss, clip, entropy, entropyTS = self.sess.run(
                    [self.train_a_op, self.a_loss, self.loss_clip, self.entropy, self.entropyTS],
                    {self.obs: s, self.actions: a, self.gaes: adv, self.term: terms,
                     self.mu: mu, self.sigma: sigma, self.e: epi})
                __, c_loss = self.sess.run([self.train_c_op, self.c_loss], {self.obs: s, self.rewards: r})
                self.logger.write_tb_log('a_loss', a_loss, self.learning_step)
                self.logger.write_tb_log('c_loss', c_loss, self.learning_step)
                self.logger.write_tb_log('clip', clip, self.learning_step)
                self.logger.write_tb_log('entropy', entropy, self.learning_step)
                self.logger.write_tb_log('entropyTS', entropyTS, self.learning_step)
                self.learning_step += 1
        else:
            for i in range(self.args['epi_train_times']):
                _, a_loss, clip, entropy, entropyTS = self.sess.run(
                    [self.train_a_op, self.a_loss, self.loss_clip, self.entropy, self.entropyTS],
                    {self.obs: s, self.actions: a, self.gaes: adv, self.term: terms,
                     self.s_a_prob: source_actor_prob, self.e: epi})
                __, c_loss = self.sess.run([self.train_c_op, self.c_loss], {self.obs: s, self.rewards: r})
                self.logger.write_tb_log('a_loss', a_loss, self.learning_step)
                self.logger.write_tb_log('c_loss', c_loss, self.learning_step)
                self.logger.write_tb_log('clip', clip, self.learning_step)
                self.logger.write_tb_log('entropy', entropy, self.learning_step)
                self.logger.write_tb_log('entropyTS', entropyTS, self.learning_step)
                self.learning_step += 1

    def load_model(self, path):
        saver = tf.train.Saver(self.policy_param)
        saver.restore(self.sess, path + ".ckpt")

    def save_model(self, path):
        saver = tf.train.Saver(self.policy_param)
        saver.save(self.sess, path + ".ckpt")
