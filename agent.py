import numpy as np

from network import build_network
from memory import Memory


class DQNAgent():

	def __init__(self, board_state):
		''' 
			Constructor for DQNAgent
			Input: board_state, (for getting the input_shape of teh network we will train)
		'''
		num_cols = board_state.shape[-1]
		self.n_actions = num_cols
		self.net = build_network(input_shape=board_state.shape, num_outputs=self.n_actions)
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

	def train(self):
		pass