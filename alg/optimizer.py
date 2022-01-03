import tensorflow as tf


class Optimizer:
    def __init__(
            self,
            optimizer,
            learning_rate,
            momentum=None
    ):
        self.opt = None
        if str(optimizer).lower() == "grad":
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif str(optimizer).lower() == "momentum":
            self.opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        elif str(optimizer).lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif str(optimizer).lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def get_optimizer(self):
        return self.opt
