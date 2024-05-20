import numpy as np
import tensorflow as tf

from aeon.distances import get_pairwise_distance_function

from sklearn.manifold import TSNE

from .utils.utils import znormalisation, accepted_layers


def get_coordinates_filters(model_paths, layer_indices, distance):
    """
    Function to get the 2D coordinates of the filters.

    Parameters
    ----------
    model_paths: list of str
        The list containing the paths of the models to be used.
    layer_indices: list of int
        The list containing the indices of the layers of each model used.
    distance: str
        The similarity measure used for the filters comparison.
        List of choices here:
        =============== ========================================
        distance        Distance Function
        =============== ========================================
        'dtw'           distances.dtw_pairwise_distance
        'shape_dtw'     distances.shape_dtw_pairwise_distance
        'ddtw'          distances.ddtw_pairwise_distance
        'wdtw'          distances.wdtw_pairwise_distance
        'wddtw'         distances.wddtw_pairwise_distance
        'adtw'          distances.adtw_pairwise_distance
        'erp'           distances.erp_pairwise_distance
        'edr'           distances.edr_pairwise_distance
        'msm'           distances.msm_pairiwse_distance
        'twe'           distances.twe_pairwise_distance
        'lcss'          distances.lcss_pairwise_distance
        'euclidean'     distances.euclidean_pairwise_distance
        'squared'       distances.squared_pairwise_distance
        'manhattan'     distances.manhattan_pairwise_distance
        'minkowski'     distances.minkowski_pairwise_distance
        'sbd'           distances.sbd_pairwise_distance
        =============== ========================================

    Returns
    -------
    list_coordinates: list
        A list containing the 2D filters coordinates of each model
        of the chosen layer.
    list_filters: list
        A list containing the original filters of each model of the
        chosen layer.
    """
    # store filters set from each model into a list
    list_filters = []
    for i in range(len(model_paths)):
        filters = _load_model_filters(model_paths[i], layer_indices[i])
        list_filters.append(filters)

    # stack all filters into a single array
    filters_array = list_filters[0]
    kernel_size = filters_array.shape[1]
    for i in range(1, len(list_filters)):
        if list_filters[i].shape[1] != kernel_size:
            print("All filters should have the same kernel size")
            exit()
        filters_array = np.vstack((filters_array, list_filters[i]))

    # get tsne coordinates
    coordinates_array = _compute_tsne_coordinates(filters_array, distance)

    # get list from stacked array
    list_coordinates = []
    cpt = 0
    for i in range(len(list_filters)):
        num_filters = list_filters[i].shape[0]
        list_coordinates.append(coordinates_array[cpt : cpt + num_filters, :])
        cpt = cpt + num_filters

    return list_coordinates, list_filters


def _load_model_filters(model_path, layer_index):

    # load model and get filters at corresponding index
    model = tf.keras.models.load_model(model_path)
    layer = model.layers[layer_index]
    if isinstance(layer, accepted_layers):
        filters = layer.get_weights()[0]
        filters = np.reshape(filters, (filters.shape[0], -1))
        filters = np.swapaxes(
            np.reshape(filters, (filters.shape[0], -1)), axis1=0, axis2=1
        )
        filters = znormalisation(filters)
    else:
        print("Layer " + layer.name + " is not a 1D convolution-based layer")
        exit()
    return filters


def _compute_tsne_coordinates(filters, distance):

    distance_matrix = _compute_distance_matrix(filters, distance)
    manifold_coordinates = TSNE(init="random", metric="precomputed").fit_transform(
        distance_matrix
    )

    return manifold_coordinates


def _compute_distance_matrix(filters, distance):

    pairwise_dist_func = get_pairwise_distance_function(distance)
    distance_matrix = pairwise_dist_func(filters)

    return distance_matrix
