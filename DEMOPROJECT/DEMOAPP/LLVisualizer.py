from keras.models import load_model
import numpy as np
import os


class LossLandscapeVisualizer:
	def __init__(self, model_path, x_path, y_path, min_v=-1.5, max_v=1.5, num=10):
		print("SAVED MODEL AT: '{}'".format(model_path))
		self.nn_model = load_model(model_path)
		self.x_test = np.load(x_path)
		self.y_test = np.load(y_path)

		self.min_v = min_v
		self.max_v = max_v
		self.num = num

	def flatten_weights(self, weights):
		flatten = list()
		for layer_weights in weights:
			flatten.append(layer_weights.flatten())

		return np.concatenate(flatten)

	def get_random_unit_vector(self, length):
		vector = np.random.normal(size=length)
		# vector = np.random.rand(length)
		magnitude = np.linalg.norm(vector)
		unit_vector = np.divide(vector, magnitude)
		return unit_vector

	def get_step_vectors(self, range1, range2, num1, num2):
		v1 = np.linspace(range1[0], range1[1], num1)
		v2 = np.linspace(range2[0], range2[1], num2)

		return v1, v2

	def get_sets_of_weights(self, unit_vector_1, unit_vector_2, steps1, steps2, weights_origin):
		v1 = unit_vector_1.reshape(unit_vector_1.shape[0], 1)
		steps_1 = steps1.reshape((1, steps1.shape[0]))

		v2 = unit_vector_2.reshape(unit_vector_2.shape[0], 1)
		steps_2 = steps2.reshape((1, steps2.shape[0]))

		v1 = np.matmul(v1, steps_1).T
		v2 = np.matmul(v2, steps_2).T

		v1_idx = np.arange(0, v1.shape[0], 1)
		v2_idx = np.arange(0, v2.shape[0], 1)

		V1_idx, V2_idx = np.meshgrid(v1_idx, v2_idx)

		V1 = np.ndarray(shape=(V1_idx.shape[0], V1_idx.shape[1], v1.shape[1]), dtype=np.float)
		for i in range(V1_idx.shape[0]):
			for j in range(V1_idx.shape[1]):
				idx = V1_idx[i, j]
				temp = v1[idx]
				V1[i, j] = temp

		V2 = np.ndarray(shape=(V2_idx.shape[0], V2_idx.shape[1], v2.shape[1]), dtype=np.float)
		for i in range(V2_idx.shape[0]):
			for j in range(V2_idx.shape[1]):
				idx = V2_idx[i, j]
				temp = v2[idx]
				V2[i, j] = temp

		result_vectors = V1 + V2
		sets_of_weights = weights_origin + result_vectors

		return sets_of_weights

	def set_weights(self, weights_vector):
		new_weights = list()
		for weights_layer in self.nn_model.get_weights():
			number_of_weights = np.prod(weights_layer.shape)

			weights_to_set = weights_vector[:number_of_weights].reshape(weights_layer.shape)
			weights_vector = weights_vector[number_of_weights:]
			new_weights.append(weights_to_set)

		self.nn_model.set_weights(new_weights)

	def evaluate_model(self, weights):
		self.set_weights(weights)
		result = self.nn_model.evaluate(self.x_test, self.y_test, verbose=0)
		return result[0]

	def get_plot_data(self, weights, s1, s2):
		X, Y = np.meshgrid(s1, s2)
		Z = np.ndarray(shape=X.shape, dtype=np.float)
		for i in range(s2.shape[0]):
			for j in range(s1.shape[0]):
				Z[i][j] = self.evaluate_model(weights[i][j])

		return X, Y, Z

	def convert_to_string(self, data):
		string = '['
		for i in range(data.shape[0]):
			arr = '['
			for j in range(data.shape[1]):
				arr += str(data[i, j]) + ','
			arr = arr[:-1] + '],'
			string += arr
		string = string[:-1] + '];'
		return string

	def process(self):
		f = self.flatten_weights(self.nn_model.get_weights())
		v1 = self.get_random_unit_vector(len(f))
		v2 = self.get_random_unit_vector(len(f))
		s1, s2 = self.get_step_vectors((self.min_v, self.max_v), (self.min_v, self.max_v), self.num, self.num)

		weights = self.get_sets_of_weights(v1, v2, s1, s2, f)
		X, Y, Z = self.get_plot_data(weights, s1, s2)

		x_string = self.convert_to_string(X)
		y_string = self.convert_to_string(Y)
		z_string = self.convert_to_string(Z)

		return {'x': x_string, 'y': y_string, 'z': z_string}


if __name__ == '__main__':

	# path = os.path.join('../', '/media/models/model_joIeLZT.h5')
	print(os.listdir('../media/models'))
	path = '..' + '/media/models/model_joIeLZT.h5'
	model = load_model(path)
