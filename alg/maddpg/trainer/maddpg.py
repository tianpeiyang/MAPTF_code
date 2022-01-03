import numpy as np
import random
import tensorflow as tf
import alg.maddpg.common.tf_util as U

from alg.maddpg.common.distributions import make_pdtype
from alg.maddpg import AgentTrainer
from alg.maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, args, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        option_ph_n = tf.placeholder(dtype=tf.float32, shape=[None, int(act_pdtype_n[p_index].param_shape()[0])], name='option_action_'+str(p_index))
        # actions = tf.placeholder(dtype=tf.float32, shape=[None, int(act_pdtype_n[p_index].param_shape()[0])], name='actions_' + str(p_index))
        term = tf.placeholder(dtype=tf.float32, shape=[None], name='term'+str(p_index))
        e = tf.placeholder(tf.float32, (), name='e'+str(p_index))
        cross_entropy = tf.placeholder(tf.float32, (), name="cross_entropy")

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        option_act_pd = act_pdtype_n[p_index].pdfromflat(option_ph_n)

        t = tf.reshape(term, shape=[-1, 1])

        cross_entropy_value = option_act_pd.kl(act_pd)
        entropyTS = tf.reduce_mean(cross_entropy_value)
        #(cross_entropy_value)
        #entropyTS = tf.reduce_sum(cross_entropy_value, axis=1,
        #                          keepdims=True)
        weight = 0.5 + tf.tanh(3 - args['c3'] * e) / 2
        entropyTS = entropyTS * weight * args['c1']
        #entropyTS = tf.reduce_mean(entropyTS)
        act = None
        act_soft_max = None

        if args['continuous_action']:
            act_sample = act_pd.sample()
            act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        else:
            act_sample = act_pd.sample()
            act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
            act_sample_soft_max = act_pd.soft_max_sample(option_act_pd, (1 - t) * weight)
            act_soft_max = U.function(inputs=[obs_ph_n[p_index]] + [option_ph_n] + [term] + [e], outputs=act_sample_soft_max)

        act_param = act_pd.params()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        p_loss = pg_loss + p_reg * 1e-3
        loss = p_loss + cross_entropy

        X = tf.distributions.Categorical(probs=act_sample)
        Y = tf.distributions.Categorical(probs=act_ph_n[p_index])
        distillation_loss = tf.distributions.kl_divergence(X, Y)

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        distillation = U.minimize_and_clip(optimizer, distillation_loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [cross_entropy], outputs=[loss, p_loss], updates=[optimize_expr])
        distillation_train = U.function(inputs=[obs_ph_n[p_index]] + [act_ph_n[p_index]], outputs=[distillation_loss],
                           updates=[distillation])
        p_distribution = U.function(inputs=[obs_ph_n[p_index]], outputs=p)
        p_distribution_params = U.function(inputs=[obs_ph_n[p_index]], outputs=act_param)
        cross_entropy_fnc = U.function(inputs=[obs_ph_n[p_index]] + [option_ph_n] + [term] + [e], outputs=entropyTS)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, distillation_train, act_soft_max, train, p_func_vars, update_target_p, p_distribution, cross_entropy_fnc, p_distribution_params, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss#+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, q_func_vars, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, use_option=False, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.use_option = use_option
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_func_vars, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args['learning_rate_c']),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args['n_layer_a_1']
        )
        self.act, self.distillation_train, self.act_soft_max, self.p_train, self.p_func_vars, self.p_update, self.p_distribution, self.cross_entropy, self.p_distribution_params,self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args['learning_rate_a']),
            args=args,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args['n_layer_a_1']
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args['batch_size'] * args['epi_step']
        self.replay_sample_index = None

    def action(self, obs, agent=None, term=None, epi=None):
        #print(self.p_distribution(obs[None])[0])
        if self.args['is_soft_max_action']:
            dis = self.distribution(agent, obs)
            return np.clip(self.act_soft_max(*([obs[None]] + [dis[None]] + [[term]] + [epi]))[0], -1, 1)
        else:
            return np.clip(self.act(obs[None])[0], -1, 1)

    def distribution(self, agent, obs):
        return agent.p_distribution(obs[None])[0]

    def choose_deterministic_action(self, obs):
        #print(self.p_distribution(obs[None])[0])
        if self.args['continuous_action']:
            return self.p_distribution_params(obs)
        else:
            return np.clip(self.act(obs), -1, 1)

    def experience(self, obs, act, rew, new_obs, option, term, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, option, term, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, e):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        agent_index = 0
        for i in range(self.n):
            if len(agents[agent_index].replay_buffer) > len(agents[i].replay_buffer):
                agent_index = i

        self.replay_sample_index = agents[agent_index].replay_buffer.make_index(self.args['batch_size'])
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        option_n = []
        term_n = []

        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, option, term, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            option_n.append(option)
            term_n.append(term)
        obs, act, rew, obs_next, o, t, done = self.replay_buffer.sample_index(index)

        options = option_n[self.agent_index]
        term = term_n[self.agent_index]
        if self.use_option or self.args['is_soft_max_action']:
            distribution_n = [agents[i].p_distribution(obs_n[agent_index]) for i in range(self.n)]
            distribution = []
            for i in range(self.args['batch_size']):
                distribution.append(distribution_n[options[i]][i])
            cross_entropy = self.cross_entropy(*([obs_n[self.agent_index]] + [distribution] + [term] + [e]))
        else:
            cross_entropy = 0

        # train q network
        num_sample = 1
        target_q = 0.0
        for k in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args['reward_decay'] * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        [loss, p_loss] = self.p_train(*(obs_n + act_n + [cross_entropy]))

        self.p_update()
        self.q_update()

        return [q_loss, loss, p_loss, cross_entropy, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

    def distillation(self, agents):
        self.replay_sample_index = agents[0].replay_buffer.make_index(self.args['batch_size'])
        # collect replay sample from all agents
        obs_n = []
        for i in range(self.n):
            obs, act, rew, obs_next, option, term, done = agents[i].replay_buffer.sample_index(self.replay_sample_index)
            obs_n.append(obs)
        action_i = [agents[k].choose_deterministic_action(obs_n[k]) for k in range(self.n)]
        for i in range(self.n):
            value_i = agents[i].q_debug['q_values'](*(obs_n + action_i))
            self.q_train(*(obs_n + action_i + [value_i]))
            self.distillation_train(*([obs_n[i]] + [action_i[i]]))

    def copy_parameter_hard(self, agents):
        for agent in agents:
            [maddpg.assign(dis) for dis, maddpg in zip(self.p_func_vars, agent.p_func_vars)]
            [maddpg.assign(dis) for dis, maddpg in zip(self.q_func_vars, agent.q_func_vars)]

