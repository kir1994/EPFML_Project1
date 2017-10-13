import numpy as np

def compute_pca(x, number_pc=-1):
	D, N = np.shape(x)

	# Center the data
	g = (1/D)*x.T.dot(np.eye(D)).dot(np.ones((D, 1))) # center of mass
	x_c = x - np.ones((D, 1)).dot(g.T)
	
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
	pc = x_c.dot(np.array(vectors[:number_pc]))
	return pc




