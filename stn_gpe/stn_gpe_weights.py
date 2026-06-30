import numpy as np


def interconnectivity(num_units:int, weights:np.ndarray):
    mid_val = num_units//2
    inter_wt = np.zeros((num_units,num_units))
    inter_wt[0:mid_val, 0:mid_val] = weights[0]
    inter_wt[mid_val:num_units,0:mid_val] = weights[1]
    inter_wt[0:mid_val,mid_val:num_units] = weights[2]
    inter_wt[mid_val:num_units, mid_val:num_units] = weights[3]

    return inter_wt

def wt_four_blocks(ep,prob,wt_shape):
    # wt = wt_four_blocks([0.1,0.1,0.1,0.1], [0.1,0.5,0.7,0.9], (256,16,16))
    weight_array = np.zeros(wt_shape)
    total_neurons = wt_shape[0]
    num_neurons_percluster = total_neurons//4 #64
    length_percluster = int(np.sqrt(num_neurons_percluster)) # 8
    cluster_shape = (num_neurons_percluster, 
                     length_percluster,
                     length_percluster)
    cluster_1 = random_wts_sparse(ep[0],prob[0],wt_shape = cluster_shape)
    cluster_2 = random_wts_sparse(ep[1],prob[1],wt_shape = cluster_shape)
    cluster_3 = random_wts_sparse(ep[2],prob[2],wt_shape = cluster_shape)
    cluster_4 = random_wts_sparse(ep[3],prob[3],wt_shape = cluster_shape)

    weight_array[0:num_neurons_percluster,0:length_percluster,0:length_percluster] = cluster_1
    weight_array[num_neurons_percluster: 2 * num_neurons_percluster,length_percluster:2 * length_percluster,0:length_percluster] = cluster_2
    weight_array[2 * num_neurons_percluster:3 * num_neurons_percluster,0:length_percluster,length_percluster: 2 * length_percluster] = cluster_3
    weight_array[3 * num_neurons_percluster:4* num_neurons_percluster,length_percluster:2*length_percluster,length_percluster:2*length_percluster] = cluster_4

    return weight_array



def random_wts_sparse(ep: float, prob:float, wt_shape: tuple):
    '''
    Function to create random weights

    Args:
        ep: epsilon value for scaling
        prob (float): probability of sparsity
        wt_shape: shape of the weight matrix required
    Returns:
        wt: randomly intialised weight matrix
    '''
    wt = np.random.rand(wt_shape[0], wt_shape[1], wt_shape[2])
    for i in range(wt_shape[0]):
      weight_mat = wt[i]
      weight_mat[weight_mat < prob] = 0

    wt = ep * wt
    return wt
