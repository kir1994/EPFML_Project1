# EPFML_Project1


def build_poly(x, d=1):
    """ Build polynomial features out of the given data, up to degree 'd' """
    # At least degree 1
    if d < 1:
        d = 1
    
    tx = np.ones(x.shape[0])
    for i in range(d):
        tx = np.c_[tx, x**(i+1)]
    return tx
