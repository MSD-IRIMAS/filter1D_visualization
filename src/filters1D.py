import numpy as np
import tensorflow as tf
from .utils.utils import *
from aeon.distances import get_pairwise_distance_function
from sklearn.manifold import TSNE


accepted_layers = (tf.keras.layers.Conv1D, tf.keras.layers.SeparableConv1D, tf.keras.layers.DepthwiseConv1D, tf.keras.layers.Conv1DTranspose)


def get_coordinates_filters(model_paths, layer_indexes, distance):

	#store filters set from each model into a list
	list_filters = []
	for i in range(len(model_paths)):
		filters = load_model_filters(model_paths[i],layer_indexes[i])
		list_filters.append(filters)

	#stack all filters into a single array
	filters_array = list_filters[0]
	kernel_size = filters_array.shape[1]
	for i in range(1,len(list_filters)):
		if list_filters[i].shape[1]!=kernel_size:
			print('All filters should have the same kernel size')
			exit()
		filters_array = np.vstack((filters_array, list_filters[i]))

	#get tsne coordinates
	coordinates_array = compute_tsne_coordinates(filters_array, distance)
	
	#get list from stacked array
	list_coordinates = []
	cpt=0
	for i in range(len(list_filters)):
		num_filters = list_filters[i].shape[0]
		list_coordinates.append(coordinates_array[cpt:cpt+num_filters,:])
		cpt=cpt+num_filters

	return list_coordinates, list_filters


def load_model_filters(model_path, layer_index):

	# load model and get filters at corresponding index
	model = tf.keras.models.load_model(model_path)
	layer = model.layers[layer_index]
	if isinstance(layer, accepted_layers):
		filters = layer.get_weights()[0]
		filters = np.reshape(filters,(filters.shape[0],-1))
		filters = np.swapaxes(np.reshape(filters,(filters.shape[0],-1)),axis1=0,axis2=1)
		filters = znormalisation(filters)
	else:
		print('Layer ' + layer.name + ' is not a 1D convolution-based layer')
		exit()
	return filters


def compute_tsne_coordinates(filters, distance):


	distance_matrix = compute_distance_matrix(filters, distance)
	manifold_coordinates = TSNE(init='random', metric='precomputed').fit_transform(distance_matrix)

	return manifold_coordinates


def compute_distance_matrix(filters, distance):

	pairwise_dist_func = get_pairwise_distance_function(distance)
	distance_matrix = pairwise_dist_func(filters)

	return distance_matrix
