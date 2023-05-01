import numpy as np

def load_karate():
    
    # load graph
    edges = []
    f = open('./karate.txt')
    for line in f.readlines():
        edge = np.array(line.replace('\n', '').split()).astype('int')
        edges.append(edge)
    f.close()

    # index from 0 instead of 1
    edges = np.array(edges) - 1

    # num vertices
    N  = np.max(edges) + 1

    # construct symmetric adjacency matrix
    X = np.zeros((N, N))
    for e0, e1 in edges:
        X[e0, e1] = 1
        X[e1, e0] = 1
        
    # load node coordinates
    coords = []
    f = open('./karatexypos.txt')
    for line in f.readlines():
        coord = np.array(line.replace('\n', '').split()).astype('float64')
        coords.append(coord)
    f.close()

    coords = np.array(coords)

        
    return X, N, edges, coords

def compute_node_and_link_counts(X, z):
    
    # Count number of nodes in each component
    m = np.sum(z, 0).T
    
    # Count number of links between components
    M = z.T@X@z - np.diag(np.sum((X@z) * z, axis=0)/2)
    
    # Count number of non-links between components
    Mbar = np.outer(m, m) - np.diag(m*(m+1)/2) - M
    
    return m, M, Mbar

