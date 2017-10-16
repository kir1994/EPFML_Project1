import numpy as np

def compute_pca(x, number_pc=-1):
    N, D = np.shape(x)

    # Center the data
    g = np.mean(x, axis=0) # center of mass
    x_c = x - np.ones((N, 1)).dot(g.reshape((1, D)))
    
    # Spectral analysis
    Sigma = x_c.T.dot(x_c)/D
    values, vectors = np.linalg.eig(Sigma)
    values, vectors = zip(*sorted(zip(values, vectors), reverse=True)) # eigenvalues and eigenvectors sorted together in decreasing order

    # Selecting the principal components
    if number_pc < 0:
        rank  = np.sum(values)
        sum_values = values[0]
        number_pc = 1
        while sum_values/rank < 0.95:
            sum_values += values[number_pc]
            number_pc += 1
    pc = x_c.dot(np.array(vectors[:number_pc]).T) + np.ones((N, 1)).dot(g[:number_pc].reshape((1, number_pc)))
    return pc, vectors[:number_pc]



