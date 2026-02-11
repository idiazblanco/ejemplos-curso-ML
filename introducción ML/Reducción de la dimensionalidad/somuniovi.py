"""
##################################################################

SOMUniOvi

    GSDPI@MIDAS, Universidad de Oviedo, 2019

    Authors:
    Ignacio Díaz Blanco    <idiaz@uniovi.es>
    Abel A. Cuadrado Vega  <cuadrado@isa.uniovi.es>
    Diego García Pérez     <diegogarcia@isa.uniovi.es>
    Ana González Muñiz     <agonzalez@isa.uniovi.es>

    If you find this work useful in your research,
    please give credits to the authors by citing:
    ¿?

##################################################################
"""


from numpy import *
from matplotlib.pyplot import clf, imshow, contour, colorbar, title, subplot, gca, tight_layout
from numpy import max, min


def matdist(Xi,Xj):
    """
    MATDIST
    -------
    d_ij = matdist(Xi,Xj)
    Calculates the distance matrix between sets of vectors

    Inputs
    ------
    Xi, Xj:  Data matrices of size (R,NXi) and (R,NXj) composed
             by column vectors of size (R,1)
    
    Output
    ------
    d_ij:    Matrix (NXi,NXj) with the euclidean distances between 
             the columns of Xi and the columns of Xj
    
    """

    if len(Xi.shape)==1:
        Xi = Xi.reshape(Xi.shape[0],1)
    
    if len(Xj.shape)==1:
        Xj = Xj.reshape(Xj.shape[0],1)
    
    xi = array([sum(Xi**2,0)]*Xj.shape[1]).T
    xj = array([sum(Xj**2,0)]*Xi.shape[1])
    d_ij  = xi + xj - 2*dot(Xi.T,Xj)

    return d_ij


def cartprod(a):
    """
    CARTPROD
    --------
    p = cartprod(a)
    Produces an array with all possible combination between the arrays 
    contained in the tuple a. It is useful to create a set of points in
    a n-dim grid (e.g. to build the pos vector in the SOM definition)

    Input
    -----
    a:  is a tuple containing arrays a = (a1,a2,...,an)

    Output
    ------
    p:  is a nD array of shape (len(a1),len(a2),...) containing 
        all possible combinations between elements of a1, a2,... 
    
    """

    temp = meshgrid(*a)
    
    p = []
    for i in map(ravel,temp):
        p.append(i)
    
    return array(p)


class SOM(object):
    """
    SOM
    ---
    Returns a ```Self-Organizing Map``` object

    Object attributes
    -----------------
    dims:    size of the lattice
    gi:      nodes in the projection space
    mi:      nodes in the input space (codebooks)
    R:       dimensionality of input space
    D:       dimensionality of output space
    S:       number of SOM units
    labels:  labels
    
    Methods
    -------
    constructor(p):  sets object attributtes according to input data
    pca_init(p):     PCA initialization
    train(p):        train the SOM
    somdist():       distance map for SOM
    planes():        plot SOM planes
    fproj(p):        forward projection using the SOM

    """

    def __init__(self, p, dims=(10,10), labels=None):
        """
    	CONSTRUCTOR
        -----------
        Sets object attributtes according to input data

        Input
        ------
        p:  numpy array of the points to be projected (R,S)
    
        """
        
        self.dims = dims
        self.gi   = cartprod(tuple(map(range,dims)))
        self.mi   = random.randn(p.shape[0],self.gi.shape[1])
        self.R    = self.mi.shape[0]
        self.D       = self.gi.shape[0]
        self.S    = self.gi.shape[1]
        self.labels = labels
        return


    def pca_init(self,p):
        """
    	PCA_INIT
        -----------
        PCA initialization

        Input
        ------
        p:  numpy array of the points to be projected (R,S)
    
        """

        u,s,v     = linalg.svd(p)

        # Compute covariance matrix
        C = cov(p)

        # Compute singular values of C
        u,s,v = linalg.svd(C)

        D  = self.D         
        gi = self.gi

        # Mean vectors for low dim lattice and input data
        gm = mean(gi,axis=1,keepdims=True)
        pm = mean(p, axis=1,keepdims=True)
        
        # Normalize gi to be centered and spanning (-1,1) at all dims
        gi_  = 2*gi/array(self.dims)[:,newaxis]-1
        
        # Initilize the codebooks using the svd approach
        # span gi_ on the directions of the principal vectors
        # with a 3*sigma span on each dimension
        self.mi = 3*(u[:,:D]*sqrt(s[:D]))@gi_ + pm


    def train(self,p,epochs=10,N_initial=None,N_final=1,verbose=True):
        """
    	TRAIN
        -----------
        Train the SOM

        Input
        ------
        p:  numpy array of the points to be projected (R,S)
    
        """
        
        if N_initial==None:
            N_initial=max(self.dims)

        Nc   = linspace(N_initial,N_final,epochs)
        for i in range(epochs):
            d = matdist(self.mi,p)
            q = d.min(0)
            c = d.argmin(0)
            hci  = exp(-matdist(self.gi[:,c],self.gi)/(2*Nc[i]))
            hcis = sum(hci,0)
            self.mi = ((hci.T@p.T)/r_[[hcis]*p.shape[0]].T).T
            if (verbose):
                print("Epoch: %d \t Neigh %f \t MSE = %f"%(i,Nc[i],mean(q)))


    def somdist(self):
        """
    	SOMDIST
        -----------
        Distance map for SOM

        """

        d = zeros(self.S)
        for i in range(self.S):
            dg   = matdist(self.gi,self.gi[:,i])
            dm   = matdist(self.mi,self.mi[:,i])
            d[i] = mean(dm[nonzero(dg<=2)])
        return d[newaxis,:] 


    def planes(self,vmin=None,vmax=None):
        """
    	PLANES
        -----------
        Plot SOM planes

        """
        ax = planes(self.mi,self.gi,self.dims,vmin=vmin,vmax=vmax,labels=self.labels)
        return ax

    
    def fproj(self,p):
        """
        FPROJ
        -----
        pr,wr,res = fproj(p)
        Forward Projection using the SOM
        
        Input
        -----
        p:    numpy array of the points to be projected (R,S)

        Outputs
        -------
        pr:   numpy array of 2D projected points (2,S)
        wr:   numpy array of best matching codebook vectors (R,S)
        res:  numpy array of residual vectors (R,S)

        """

        d = matdist(self.mi,p)
        c = d.argmin(0)
        pr = self.gi[:,c]
        wr = self.mi[:,c]
        res = p - wr
        
        return pr,wr,res,c


def planes(mi,gi,dims,vmin=None,vmax=None,labels=None):
    """
    PLANES
    ------
    planes(mi,gi,dims,vmin=None,vmax=None,labels=None)
    Plot SOM planes
    
    Inputs
    ------
    mi:    nodes in the input space (codebooks)
    gi:    nodes in the projection space
    dims:  size of the lattice

    """

    clf();
    N = mi.shape[0]

    # Handle colormap limits
    if (vmax != None):
        if (isscalar(vmax)):
            vmax = ones(N)*vmax
    else:
        vmax = max(mi,axis=1)


    if (vmin != None):
        if (isscalar(vmin)):
            vmin = ones(N)*vmin
    else:
        vmin = min(mi,axis=1)


    # Determine a balanced distribution of subfigures
    nx = int(round(sqrt(N)))
    ny = nx
    while nx*ny < N:
        ny += 1
    
    ax = []    
    for i in range(mi.shape[0]):
        subplot(nx,ny,i+1)

        z = reshape(mi[i,:],dims)
        contour(z,30,origin='lower', extent=(min(gi[0,:]),max(gi[0,:]),min(gi[1,:]),max(gi[1,:])))
        imshow(z,origin='lower', extent=(min(gi[0,:]),max(gi[0,:]),min(gi[1,:]),max(gi[1,:])),vmin=vmin[i],vmax=vmax[i])
        colorbar()
        if not (labels==None):
            title(labels[i])

        ax.append(gca())    
    
    tight_layout()
    
    return ax



