import networkx as nx
from scipy.io import loadmat
import numpy as np
import scipy.sparse
from scipy.spatial import distance_matrix
import warnings


def load_graph(graph_id, normalize_pos=True, path='./'):
    
    print('\nLoading the graph: ' + graph_id)
    # Turn off SparseEfficiencyWarning: this is raised as we try to fill
    # values of a csr sparse matrix. Since we need to fill a few values,
    # this is way more efficient than converting the matrix to a lil format.
    warnings.filterwarnings('ignore', category=scipy.sparse.SparseEfficiencyWarning)
    
    if graph_id == 'minnesota':
        # Load the data
        raw_data = loadmat(path + 'data/data_minnesota.mat')
        # Add a missing edge to the adjacency matrix
        raw_data['A'][349, 355] = 1
        raw_data['A'][355, 349] = 1
        # Create the graph from the adjacency matrix        
        G = nx.from_scipy_sparse_matrix(raw_data['A'])
        # Define the nodes' positions
        nodes = raw_data['xy']
        
    elif graph_id == 'bunny':
        # Load the data
        raw_data = loadmat(path + 'data/data_bunny.mat')
        # Project the 3d points to 2d
        nodes = raw_data['bunny'][:, :2]
        # Remove close nodes 
        threshold = 0.0025
        nodes = incremental_thinning(nodes, threshold)
        # Generate a nearest neighbor graph with radius r    
        r = 0.01
        G = nn_graph(nodes, r)
       
    elif graph_id == '2moon':
        filename = 'data_2moon.mat'
        r = 0.5
        # Load the data
        raw_data = loadmat(path + 'data/' + filename)
        # Generate a nearest neighbor graph with radius r    
        G = nn_graph(raw_data['nodes'], r)        
        # Define the nodes' positions
        nodes = raw_data['nodes']
        
    elif graph_id == 'emptyset':
        filename = 'data_emptyset.mat'
        r = 0.2
        # Load the data
        raw_data = loadmat(path + 'data/' + filename)
        # Generate a nearest neighbor graph with radius r    
        G = nn_graph(raw_data['nodes'], r)       
        # Define the nodes' positions
        nodes = raw_data['nodes']

    elif graph_id == 'sensor1':
        filename = 'data_sensor1.mat'
        r = 1/6
        # Load the data
        raw_data = loadmat(path + 'data/' + filename)
        # Generate a nearest neighbor graph with radius r    
        G = nn_graph(raw_data['nodes'], r)        
        # Define the nodes' positions
        nodes = raw_data['nodes']

    elif graph_id == 'sensor2':
        filename = 'data_sensor2.mat'
        r = 1/6
        # Load the data
        raw_data = loadmat(path + 'data/' + filename)
        # Generate a nearest neighbor graph with radius r    
        G= nn_graph(raw_data['nodes'], r)        
        # Define the nodes' positions
        nodes = raw_data['nodes']

    elif graph_id == 'star':
        num_nodes = 40
        # The other nodes are on the unit circle
        t = (2 * np.pi / (num_nodes - 1) * np.arange(num_nodes - 1))[:, None]
        nodes = np.c_[np.cos(t), np.sin(t)]
        # The first node is in zero
        nodes = np.r_[nodes, np.zeros((1, 2))]
        # Generate the adjacency matrix: all nodes are connected to the center
        r1 = (num_nodes - 1) * np.ones(num_nodes)
        r2 = np.arange(num_nodes)
        row_ind = np.r_[r1, r2]
        col_ind = np.r_[r2, r1]
        data = np.ones(len(row_ind))
        A = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))
        # Create the graph from the adjacency
        G = nx.from_scipy_sparse_matrix(A)

    elif graph_id == 'rand':
        num_nodes = 300
        # Generate num_nodes random points in [0, 1]^2
        nodes = np.random.rand(num_nodes, 2)
        # Remove close nodes 
        threshold = 0.02
        nodes = incremental_thinning(nodes, threshold)
        # Generate a nearest neighbor graph with radius r    
        r = 1/6
        G = nn_graph(nodes, r)

    elif graph_id == 'rand_sparse':
        num_nodes = 100000
        num_edges = 2 * num_nodes
        # Generate num_nodes random points in [0, 1]^2
        nodes = np.random.rand(num_nodes, 2)
        # Generate a random adjacency matrix
        r1 = np.random.randint(0, num_nodes, int(np.floor(num_edges / 2)))
        r2 = np.random.randint(0, num_nodes, int(np.floor(num_edges / 2)))
        row_ind = np.r_[r1, r2]
        col_ind = np.r_[r2, r1]
        data = np.ones(len(row_ind))
        A = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))
        # Symmetrize the matrix
        A.setdiag(0)
        A.eliminate_zeros()
        # Create the graph from the adjacency
        G = nx.from_scipy_sparse_matrix(A)
            
    # Assign the nodes' positions
    if normalize_pos:
        nodes = normalize(nodes)
    # Create a dictionary of positions, as node_id: (x, y)
    pos = {idx: nodes[idx] for idx in range(len(nodes))}
    # Assign the node position as a node's attribute
    nx.set_node_attributes(G, pos, 'pos')

    print('\t\t|_ Done!')

    return G


def normalize(pos):
    pos = (pos - np.min(pos, axis=0)) / (np.max(pos, axis=0) - np.min(pos, axis=0))
    return pos


def incremental_thinning(all_nodes, threshold):
    nodes = np.empty((0, 2))
    while len(all_nodes):
        current_node = np.atleast_2d(all_nodes[0])
        nodes = np.r_[nodes, current_node]
        d = distance_matrix(current_node, all_nodes[0:])
        idx_remove = np.argwhere(d <= threshold)
        all_nodes = np.delete(all_nodes, idx_remove, axis=0)
    return nodes


def nn_graph(nodes, r):
    d = distance_matrix(nodes, nodes)
    d[d > r] = 0
    d[d > 0] = 1
    G = nx.from_numpy_matrix(d)
    return G


def sim_graph(nodes):
    A = 0
    
    return A
    
    
    
    