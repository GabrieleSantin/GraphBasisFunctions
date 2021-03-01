import numpy as np
from graph_loaders import load_graph
from joblib import Parallel, delayed

def IC(G,S,p,mc):
    """
    Inputs: G: nx.graph object
            S:  List of seed nodes
            p:  Propagation probability
            mc: Number of Monte-Carlo simulations,
    Output: Average number of nodes influenced by seed nodes in S
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        
        # Simulate the propagation process      
        new_active, A = S[:], S[:]
        while new_active:
                       
            # 1. Find out-neighbors for each newly active node
            targets = []
            for node in new_active:
                targets += G.neighbors(node)
    
            # 2. Determine newly activated neighbors (set seed and sort for consistency)
            np.random.seed(i)
            success = np.random.uniform(0,1,len(targets)) < p
            new_ones = list(np.extract(success, sorted(targets)))
            
            # 3. Find newly activated nodes and add to the set of activated nodes
            new_active = list(set(new_ones) - set(A))
            A += new_active
            
        spread.append(len(A))
        
    return np.mean(spread), A


def ICgreedy(G, num_nodes, p, mc):

    # Initialize the list of influent nodes
    idxIC = []
    # Inizialize their spread
    spread = []

    # Get a list of the nodes
    all_nodes = [int(n) for n in range(len(G))]

    # Repeat to select the first num_nodes nodes
    for i in range(num_nodes):
        # Initially the list of nodes infected by each candidate influencer
        loc_spread = []

        # Run over each node of G
        for j in all_nodes:
            # Initialize the spreading nodes to the influent nodes
            idxQ = idxIC[:]
            # Add the current node if not present
            if j not in idxIC:
                idxQ.append(j)
            # Spread starting from idxQ and compute the mean num of infected 
            mean_num_infected = IC(G, idxQ, p, mc)[0]
            # Save the result to the list at the nodes' position
            loc_spread.append(mean_num_infected)

        
        # Find the value of largest spread    
        max_value = max(loc_spread)
        # Save the len of the largest spread
        spread.append(max_value)
        # Find the node that caused the largest spread
        idx = loc_spread.index(max_value)
        # Add this node to list of influent nodes
        idxIC.append(all_nodes[idx])  
        # Remove this node from the list
        all_nodes.pop(idx)
        print('MI nodes: ', idxIC)

    return idxIC, spread


def demo():
    # Create Graph
    G = load_graph('sensor1')
    
    # Choose parameters for test run
    S = [10,2,12]   # List of seed nodes for 1 test run
    K = 4          # Number of most influencing nodes
    p = 0.2         # Propagation probability
    mc = 500        # Number of Monte-Carlo simulations
    
    # Run 1 test run with IC
    output_nx, A_nx = IC(G,S,p,mc)
    
    # Run entire ICgreedy on graph
    idxIC, spread = ICgreedy(G,K,p,mc)


if __name__ == '__main__':
    demo()
