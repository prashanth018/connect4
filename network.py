import tensorflow as tf
from tensorflow import keras


def build_network(input_shape, num_outputs):
	
	model = keras.models.Sequential()
	model.add(keras.layers.InputLayer(input_shape=input_shape))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(32))
	model.add(keras.layers.LeakyReLU(alpha=0.02))
	model.add(keras.layers.Dense(num_outputs))
	return model


if __name__ == '__main__':

	import numpy as np

	board_state = np.random.randint(-1, 2, size=(6, 7))
	num_outputs = 7

	sample_network = build_network(board_state.shape, num_outputs)
	print(sample_network.predict( np.expand_dims(board_state, axis=0) ))

	board_state1 = np.random.randint(-1, 2, size=(6, 7))
	board_state2 = np.random.randint(-1, 2, size=(6, 7))

	assert ( sample_network.predict( np.expand_dims(board_state1, axis=0) ) != \
		   	 sample_network.predict( np.expand_dims(board_state2, axis=0) ) ).all()