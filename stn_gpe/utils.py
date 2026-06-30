import yaml
import numpy as np

def load_yaml(yaml_file):
    ''' 
    Function to load yaml file

    Args:
        yaml_file (str): File path 
    Returns:
        data (dict): Returns loaded yaml file as dictionary
    '''
    with open(yaml_file,'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, filename):
    with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


def lfp_dist_matrix(n:int, xc:int, yc:int)->np.ndarray:
    '''
    Computes 1/dist from a mentioned x,y coordinates

    Args:
        n (int): grid size
        xc (int): x coordinate of center
        yc (int): y coordinate of center

    returns:
        w (np.ndarray): 1/dist matrix
    '''
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if np.sqrt((i-xc)**2 + (j-yc)**2) == 0:
                w[i][j] = 1
            else:
                w[i][j] = 1/(np.sqrt((i-xc)**2 + (j-yc)**2))

    return w 