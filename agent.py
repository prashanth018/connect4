import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from network import build_network
from memory import Memory


class DQNAgent():

	def __init__(self, board_state, batch_size=32):
		''' 
			Constructor for DQNAgent
			Input: board_state, (for getting the input_shape of teh network we will train)
		'''
		num_cols = board_state.shape[-1]
		self.n_actions = num_cols
		
		self.net = build_network(input_shape=board_state.shape, num_outputs=self.n_actions)
		self.target_net = build_network(input_shape=board_state.shape, num_outputs=self.n_actions)
		self.target_net.set_weights(self.net.get_weights())
		self.target_net.trainable = False
		self.net_optimizer = tf.train.AdamOptimizer()

		self.memory = Memory(batch_size)

		self.curr_obs = None
		self.curr_action = None


	def get_action(self, board_state, epsilon=0.05):
		'''
			env will ask agent to give an action by passing it current obs.
			agent acts epsilon-greedily
		'''
		action = None
		if np.random.uniform() < epsilon:
			action = np.random.randint(low=0, high=self.n_actions)
		q_vals = self.net.predict( np.expand_dims(board_state, axis=0) )[0]
		action = np.argmax( q_vals )
		self.curr_action = action
		return self.curr_action


	def receive_next_obs_rew_done(self, next_obs, reward, done):
		if self.curr_obs is not None:
			self.memory.add( curr_obs=self.curr_obs,
							 curr_action=self.curr_action,
							 next_obs=next_obs,
							 reward=reward,
							 done=done )
		if not done:
			self.curr_obs = next_obs
		else:
			self.curr_obs = None
			self.curr_action = None


	def train_one_batch(self, gamma):
		curr_obs, curr_actions, next_obs, rewards, dones = self.memory.sample()
		curr_actions = tf.one_hot(curr_actions, depth=self.n_actions, dtype=tf.int32)
		next_qvals = tf.max(self.target_net.predict(next_obs), axis=1)
		targets = next_qvals * gamma + rewards
		
		with tf.GradientTape() as t:
			curr_qvals = self.net.predict(curr_obs)
			curr_qvals = tf.sum(curr_actions * curr_qvals, axis=1)
			loss = tf.reduce_mean((targets - curr_qvals) ** 2)

		grad = t.gradient(loss, net.trainable_variables)
		self.net_optimizer.apply_gradients(zip(grads, self.net.trainable_variables))


	def train(self, gamma=0.95, num_epochs=1000):
		losses = []
		for epoch in range(num_epochs):
			loss = self.train_one_batch(gamma)
			losses.append(loss)
		self.target_net.set_weights(self.net.get_weights())
		return np.mean(losses)



if __name__ == '__main__':

	board_state = np.random.randint(-1, 2, size=(6, 7))
	agent = DQNAgent(board_state)