'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent


from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete

def observation_input(ob_space, name='Ob', batch_size=None, scale=False):
    """
    Build observation input with encoding depending on the observation space type

    When using Box ob_space, the input will be normalized between [1, 0] on the bounds ob_space.low and ob_space.high.

    :param ob_space: (Gym Space) The observation space
    :param name: (str) tensorflow variable name for input placeholder
    :param scale: (bool) whether or not to scale the input
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
    if isinstance(ob_space, Discrete):
        observation_ph = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_observations = tf.cast(tf.one_hot(observation_ph, ob_space.n), tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, Box):
        observation_ph = tf.placeholder(shape=(batch_size,) +ob_space.shape, dtype=ob_space.dtype, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        # rescale to [1, 0] if the bounds are defined
        if (scale and
           not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
           np.any((ob_space.high - ob_space.low) != 0)):

            # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
            processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiBinary):
        observation_ph = tf.placeholder(shape=(ob_space.n,), dtype=tf.int32, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        processed_observations = tf.expand_dims(processed_observations, axis=0)
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiDiscrete):
        observation_ph = tf.placeholder(shape=(len(ob_space.nvec),), dtype=tf.int32, name=name)
        processed_observations = tf.expand_dims(observation_ph, axis=0)
        processed_observations = tf.concat([
            tf.cast(tf.one_hot(input_split, ob_space.nvec[i]), tf.float32) for i, input_split
            in enumerate(tf.split(processed_observations, len(ob_space.nvec), axis=-1))
        ], axis=-1)
        return observation_ph, processed_observations

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(ob_space).__name__))


class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.hidden_size = hidden_size

        # Build graph
        self.generator_obs_ph, generator_obs = observation_input(env.observation_space, name="generator_obs")
        self.generator_acs_ph, generator_acs = observation_input(env.action_space, name="generator_actions")
        self.expert_obs_ph, expert_obs = observation_input(env.observation_space, name="expert_obs")
        self.expert_acs_ph, expert_acs = observation_input(env.action_space, name="expert_actions")
        generator_logits = self.build_graph(generator_obs, generator_acs, reuse=False)
        expert_logits = self.build_graph(expert_obs, expert_acs, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy

        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss

        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                                       self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
