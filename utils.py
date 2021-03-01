import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse
from external_algorithms import IC

        
def plot_graph(G, ax, values=[], 
               cb_label='', 
               nodelist=[],
               log_scale=False, 
               fontsize=16, nodesize=8, edgewidth=0.1,
               min_clip=1e-15, max_clip=1e15,
               show_cb=True,
               max_nodes=10000):

    if type(values) == scipy.sparse.csr_matrix:
        values = values.todense()
    
    # Set the colormap
    cmap = plt.cm.Spectral 
    # cmap = plt.cm.copper_r
    
    # Plot the nodes
    if list(values):
        if log_scale and np.min(values) == 0:
            values = np.clip(values, min_clip, max_clip)
        col = list(values)
    else:
        col = []

    nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G, 'pos'), ax=ax,
                           node_size=nodesize,
                           node_color=col,
                           cmap=cmap)
    
    # Plot the edges
    nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'), ax=ax,
                           width=edgewidth)
    
    # If required, highlight some nodes
    if list(nodelist):
        nodelist = np.ravel(nodelist)
        nodes_plot = nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G, 'pos'), ax=ax,
                               nodelist=nodelist,
                               node_shape='o',
                               node_size=5*nodesize,
                               alpha=1)
        nodes_plot.set_facecolor('none')
        nodes_plot.set_edgecolor('k')
        
    # Customize the colormap and the colorbar (if any)
    if list(values) and show_cb:
        vmin = min(values)
        vmax = max(values)
        if log_scale:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_label(cb_label, fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
    
    # Set square axis
    ax.axis('equal')     


def graph_spectrum(G, stype='ascend'):
    if G:
        # Compute the graph laplacian
        Lap = nx.linalg.normalized_laplacian_matrix(G)
        # Compute the SVD
        U, L, _ = np.linalg.svd(Lap.todense())
        # Reverse the order of L if required
        if stype == 'descend':
            idx_sort = np.argsort(L)
            L = L[idx_sort]
            U = U[:, idx_sort]
        # Normalize U to have positive first row
        signs = np.sign(np.asarray(U[0])[0])
        signs[signs==0] = 1
        U = U @ np.diag(signs)
    else:
        U, L = [], []
    return U, L


def independent_cascade_scorer(G, idx_nodes, p, mc):
   spread = []
   for idx in range(len(idx_nodes)):
       spread.append(IC(G, idx_nodes[:idx+1], p, mc)[0])
   missing_spread = np.array([(len(G) - s) / len(G) for s in spread])     
   return missing_spread


def degree_scorer(G, idx_nodes):
   degrees = []
   for idx in range(len(idx_nodes)):
       degrees.append(G.degree(idx_nodes[idx]))
   return degrees