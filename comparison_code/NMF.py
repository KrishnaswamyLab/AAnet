# This file provides the functions used in implementing the proposed method for Non-negative matrix factorization in the paper, "Non-negative Matrix Factorization via Archetypal Analysis".


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls 
from scipy.spatial import ConvexHull
import time
import timeit

def D_distance (H1, H2):

# This function computes the 'L'-distance between the two set of vectors collected in the rows of H1 and H2. In our paper notation, this is $\mathscr{L}(H_1, H_2)$.
    H1 = np.array(H1)
    H2 = np.array(H2)
    n1 = H1.shape[0]
    n2 = H2.shape[0]
    D = 0
    for i in range (0,n1):
        d = (np.linalg.norm(H1[i,:] - H2[0,:]))**2
        for j in range (1,n2):
            d = min(d, (np.linalg.norm(H1[i,:] - H2[j,:])**2))
        D = D+d
    return D

def generate_weights( n, r, alpha, n_f, deg_prob):

# This function generates 'n' weight vectors in r-dimensions, distributed as Dirichlet(alpha, alpha, ..., alpha). 'n_f' is the number of weight vector which have zero components (induce points that lie on the faces) and 'deg_prob' is the distribution of the support size of these weight vectors. Namely, these weight vectors are distributed as Dirichlet over the set of nonzero entries which is a uniformly distributed set with a size randomly generated according to 'deg_prob'.

    W = np.zeros((n,r))
    for i in range (0,n_f):
        deg_cdf = np.cumsum(deg_prob)
        t = np.random.uniform(0,1)
        ind = np.nonzero(deg_cdf>t)
        deg = np.min(ind)+1
        dirich_param = alpha*np.ones(deg)
        w = np.random.dirichlet(dirich_param)
        vertices = np.random.permutation(r)
        vertices = vertices[0:deg]
        W[i,vertices] = np.random.dirichlet(dirich_param)
    
    for i in range(n_f,n):
        dirich_param = alpha*np.ones(r)
        W[i,:] = np.random.dirichlet(dirich_param)
    return W

def l2distance( x, U, x0 ):

# This function computes <x-x0, (U^T*U)*(x-x0)>.

    lx = np.linalg.norm(x-x0)**2
    lpx = np.linalg.norm(np.dot(U,x-x0))**2
    return (lx-lpx)

def plot_H( H, col, type ):

# This function plots the 'archetypes', (rows of 'H', when they are 2-dimensional) in 'col' color using 'type' as plot options.

    v0 = H[:,0]
    v0 = np.append(v0,H[0,0])
    v1 = H[:,1]
    v1 = np.append(v1,H[0,1])
    hplt, = plt.plot(v0,v1,type,color=col,markersize=8,linewidth=3)
    return hplt 
    
def plot_data( X, col ):

# This function plots the 'data points', (rows of 'X', when they are 2-dimensional) in 'col' color.
 
    plt.plot(X[:,0],X[:,1],'o',color=col,markersize=5)

def initH( X, r ):

# This function computes 'r' initial archetypes given rows of 'X' as the data points. The method used here is the successive projections method explained in the paper.

    n = X.shape[0]
    d = X.shape[1]
    H = np.zeros((r,d))
    maxd = np.linalg.norm(X[0,:])
    imax = 0
    for i in range(1,n):
        newd = np.linalg.norm(X[i,:]) 
        if (newd>maxd):
            imax = i
            maxd = newd
    H[0,:]= X[imax,:]
    maxd = np.linalg.norm(X[0,:]-H[0,:])
    imax = 0
    for i in range(1,n):
        newd = np.linalg.norm(X[i,:]-H[0,:]) 
        if (newd>maxd):
            imax = i
            maxd = newd
    H[1,:]= X[imax,:]
    
    for k in range(2,r):
        M = H[1:k,:]-np.outer(np.ones(k-1),H[0,:])
        [U, s, V] = np.linalg.svd(M,full_matrices=False)
        maxd = l2distance(X[0,:],V,H[0,:])
        imax = 0
        for i in range(1,n):
            newd = l2distance(X[i,:],V,H[0,:]) 
            if (newd>maxd):
                imax = i
                maxd = newd
        H[k,:]= X[imax,:]
    return H

def project_simplex( x ):

# This function computes the euclidean projection of vector 'x' onto the standard simplex.

    n = len(x)
    xord = -np.sort(-x)
    sx = np.sum(x)
    lam = (sx-1.)/n
    if (lam<=xord[n-1]):
        return (x-lam)
    k = n-1
    flag = 0
    while ((flag==0)and(k>0)):
        sx -= xord[k]
        lam = (sx-1.)/k
        if ((xord[k]<=lam)and(lam<=xord[k-1])):
            flag = 1
        k -= 1
    return np.fmax(x-lam,0)

def project_principal(X,r):

# This function computes the rank 'r' pca estimate of columns of 'X'.

    U, s, V = np.linalg.svd(X)
    V = V[0:r,:]
    U = U[:,0:r]
    s = s[0:r]
    proj_X = np.dot(U, np.dot(np.diag(s), V))
    return proj_X

def prune_convex(X):

# This function output the rows of 'X' which do not lie on the convex hull of the other rows.

    n = X.shape[0]
    indices = []
    d = X.shape[1]
    pruned_X = np.empty((0,d), int)
    for i in range(0,n-1):
        print(i)
        c = np.zeros(n-1)
        AEQ = np.delete(X,i,0)
        AEQ = np.transpose(AEQ)
        AEQ = np.vstack([AEQ, np.ones((1,n-1))])
        BEQ = np.concatenate((X[i,:],[1]),0)
        res = linprog(c, A_ub=-1*np.identity(n-1) , b_ub=np.zeros((n-1,1)), A_eq=AEQ, b_eq=np.transpose(BEQ),options={"disp": True})
        if (res.status == 2):          
            pruned_X = np.append(pruned_X, X[i,:].reshape(1,d), axis=0)
            indices = np.append(indices,i)
    return [indices.astype(int),pruned_X]

# project onto a line-segment
def proj_line_seg(X, x0):

# This function computes the projection of the point x0 onto the line segment between the points x1 and x2.

    x1 = X[:,0]
    x2 = X[:,1]
    alpha = float(np.dot(np.transpose(x1-x2), x0-x2))/(np.dot(np.transpose(x1-x2), x1-x2))
    alpha = max(0,min(1,alpha))
    y = alpha*x1 + (1-alpha)*x2
    theta = np.array([alpha, 1-alpha])
    return [theta, y]

# project onto a triangle
def proj_triangle(X, x0):

# This function computes the projection of the point x0 onto the triangle with corners specified with the rows of X.

    d = len(x0)
    XX = np.zeros((d,2))
    XX[:,0] = X[:,0] - X[:,2]
    XX[:,1] = X[:,1] - X[:,2] 
    P = np.dot(np.linalg.inv(np.dot(np.transpose(XX),XX)),np.transpose(XX))
    theta = np.append(np.dot(P, x0-X[:,2]), 1-np.sum(np.dot(P, x0-X[:,2])))
    y = np.dot(X,theta)
    if ((any(theta<0)) or (any(theta>1)) or (np.sum(theta)!=1)):
        d1 = np.linalg.norm(X[:,0] - y)
        d2 = np.linalg.norm(X[:,1] - y)
        d3 = np.linalg.norm(X[:,2] - y)
        theta4,y4 = proj_line_seg(X[:,[0,1]],y) 
        d4 = np.linalg.norm(y-y4)
        theta5,y5 = proj_line_seg(X[:,[0,2]],y)
        d5 = np.linalg.norm(y-y5)
        theta6,y6 = proj_line_seg(X[:,[1,2]],y)
        d6 = np.linalg.norm(y-y6)
        d = min(d1,d2,d3,d4,d5,d6)
        if (d1 == d):
            y = X[:,0]
            theta = np.array([1,0,0])
        elif (d2 == d):
            y = X[:,1]
            theta = np.array([0,1,0])
        elif (d3 == d):
            y = X[:,2]
            theta = np.array([0,0,1])
        elif (d4 == d):
            y = y4
            theta = np.zeros(3)
            theta[[0,1]] = theta4
        elif (d5 == d):
            y = y5
            theta = np.zeros(3)
            theta[[0,2]] = theta5
        else:
            y = y6
            theta = np.zeros(3)
            theta[[1,2]] = theta6
    return[theta,y]

# project onto a tetrahedron
def proj_tetrahedron(X, x0):

# This function computes the projection of the point x0 onto the tetrahedron with corners specified with the rows of X.

    d = len(x0)
    XX = np.zeros((d,3))
    XX[:,0] = X[:,0] - X[:,3]
    XX[:,1] = X[:,1] - X[:,3]
    XX[:,2] = X[:,2] - X[:,3]
    P = np.dot(np.linalg.inv(np.dot(np.transpose(XX),XX)),np.transpose(XX))
    theta = np.append(np.dot(P, x0-X[:,3]), 1-np.sum(np.dot(P, x0-X[:,3])))
    y = np.dot(X,theta)
    if ((any(theta<0)) or (any(theta>1)) or (np.sum(theta)!=1)):
        d1 = np.linalg.norm(X[:,0] - y)
        d2 = np.linalg.norm(X[:,1] - y)
        d3 = np.linalg.norm(X[:,2] - y)
        d4 = np.linalg.norm(X[:,3] - y)
        theta5,y5 = proj_line_seg(X[:,[0,1]],y) 
        d5 = np.linalg.norm(y-y5)
        theta6,y6 = proj_line_seg(X[:,[0,2]],y)
        d6 = np.linalg.norm(y-y6)
        theta7,y7 = proj_line_seg(X[:,[0,3]],y)
        d7 = np.linalg.norm(y-y7)
        theta8,y8 = proj_line_seg(X[:,[1,2]],y) 
        d8 = np.linalg.norm(y-y8)
        theta9,y9 = proj_line_seg(X[:,[1,3]],y) 
        d9 = np.linalg.norm(y-y9)
        theta10,y10 = proj_line_seg(X[:,[2,3]],y) 
        d10 = np.linalg.norm(y-y10)
        theta11,y11 = proj_triangle(X[:,[0,1,2]],y)
        d11 = np.linalg.norm(y-y11)
        theta12,y12 = proj_triangle(X[:,[0,1,3]],y)
        d12 = np.linalg.norm(y-y12)
        theta13,y13 = proj_triangle(X[:,[0,2,3]],y)
        d13 = np.linalg.norm(y-y13)
        theta14,y14 = proj_triangle(X[:,[1,2,3]],y)
        d14 = np.linalg.norm(y-y14)
        d = min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14)
        if (d1 == d):
            y = X[:,0]
            theta = np.array([1,0,0,0])
        elif (d2 == d):
            y = X[:,1]
            theta = np.array([0,1,0,0])
        elif (d3 == d):
            y = X[:,2]
            theta = np.array([0,0,1,0])
        elif (d4 == d):
            y = X[:,3]
            theta = np.array([0,0,0,1])
        elif (d5 == d):
            y = y5
            theta = np.zeros(4)
            theta[[0,1]] = theta5
        elif (d6 == d):
            y = y6
            theta = np.zeros(4)
            theta[[0,2]] = theta6
        elif (d7 == d):
            y = y7
            theta = np.zeros(4)
            theta[[0,3]] = theta7
        elif (d8 == d):
            y = y8
            theta = np.zeros(4)
            theta[[1,2]] = theta8
        elif (d9 == d):
            y = y9
            theta = np.zeros(4)
            theta[[1,3]] = theta9
        elif (d10 == d):
            y = y10
            theta = np.zeros(4)
            theta[[2,3]] = theta10
        elif (d11 == d):
            y = y11
            theta = np.zeros(4)
            theta[[0,1,2]] = theta11
        elif (d12 == d):
            y = y12
            theta = np.zeros(4)
            theta[[0,1,3]] = theta12
        elif (d13 == d):
            y = y13
            theta = np.zeros(4)
            theta[[0,2,3]] = theta13
        else:
            y = y14
            theta = np.zeros(4)
            theta[[1,2,3]] = theta14
    return[theta,y]

# project onto a 5cell
def proj_5cell(X, x0):

# This function computes the projection of the point x0 onto the 5-cell with corners specified with the rows of X.

    d = len(x0)
    XX = np.zeros((d,4))
    XX[:,0] = X[:,0] - X[:,4]
    XX[:,1] = X[:,1] - X[:,4]
    XX[:,2] = X[:,2] - X[:,4]
    XX[:,3] = X[:,3] - X[:,4]
    P = np.dot(np.linalg.inv(np.dot(np.transpose(XX),XX)),np.transpose(XX))
    theta = np.append(np.dot(P, x0-X[:,4]), 1-np.sum(np.dot(P, x0-X[:,4])))
    y = np.dot(X,theta)
    if ((any(theta<0)) or (any(theta>1)) or (np.sum(theta)!=1)):
        d1 = np.linalg.norm(X[:,0] - y)
        d2 = np.linalg.norm(X[:,1] - y)
        d3 = np.linalg.norm(X[:,2] - y)
        d4 = np.linalg.norm(X[:,3] - y)
        d5 = np.linalg.norm(X[:,4] - y)
        theta6,y6 = proj_line_seg(X[:,[0,1]],y) 
        d6 = np.linalg.norm(y-y6)
        theta7,y7 = proj_line_seg(X[:,[0,2]],y)
        d7 = np.linalg.norm(y-y7)
        theta8,y8 = proj_line_seg(X[:,[0,3]],y)
        d8 = np.linalg.norm(y-y8)
        theta9,y9 = proj_line_seg(X[:,[0,4]],y) 
        d9 = np.linalg.norm(y-y9)
        theta10,y10 = proj_line_seg(X[:,[1,2]],y) 
        d10 = np.linalg.norm(y-y10)
        theta11,y11 = proj_line_seg(X[:,[1,3]],y) 
        d11 = np.linalg.norm(y-y11)
        theta12,y12 = proj_line_seg(X[:,[1,4]],y) 
        d12 = np.linalg.norm(y-y12)
        theta13,y13 = proj_line_seg(X[:,[2,3]],y) 
        d13 = np.linalg.norm(y-y13)
        theta14,y14 = proj_line_seg(X[:,[2,4]],y) 
        d14 = np.linalg.norm(y-y14)
        theta15,y15 = proj_line_seg(X[:,[3,4]],y) 
        d15 = np.linalg.norm(y-y15)
        theta16,y16 = proj_triangle(X[:,[0,1,2]],y)
        d16 = np.linalg.norm(y-y16)
        theta17,y17 = proj_triangle(X[:,[0,1,3]],y)
        d17 = np.linalg.norm(y-y17)
        theta18,y18 = proj_triangle(X[:,[0,1,4]],y)
        d18 = np.linalg.norm(y-y18)
        theta19,y19 = proj_triangle(X[:,[0,2,3]],y)
        d19 = np.linalg.norm(y-y19)
        theta20,y20 = proj_triangle(X[:,[0,2,4]],y)
        d20 = np.linalg.norm(y-y20)
        theta21,y21 = proj_triangle(X[:,[0,3,4]],y)
        d21 = np.linalg.norm(y-y21)
        theta22,y22 = proj_triangle(X[:,[1,2,3]],y)
        d22 = np.linalg.norm(y-y22)
        theta23,y23 = proj_triangle(X[:,[1,2,4]],y)
        d23 = np.linalg.norm(y-y23)
        theta24,y24 = proj_triangle(X[:,[1,3,4]],y)
        d24 = np.linalg.norm(y-y24)
        theta25,y25 = proj_triangle(X[:,[2,3,4]],y)
        d25 = np.linalg.norm(y-y25)
        theta26,y26 = proj_tetrahedron(X[:,[0,1,2,3]],y)
        d26 = np.linalg.norm(y-y26)
        theta27,y27 = proj_tetrahedron(X[:,[0,1,2,4]],y)
        d27 = np.linalg.norm(y-y27)
        theta28,y28 = proj_tetrahedron(X[:,[0,1,3,4]],y)
        d28 = np.linalg.norm(y-y28)
        theta29,y29 = proj_tetrahedron(X[:,[0,2,3,4]],y)
        d29 = np.linalg.norm(y-y29)
        theta30,y30 = proj_tetrahedron(X[:,[1,2,3,4]],y)
        d30 = np.linalg.norm(y-y30)
        d = min(d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30)
        if (d1 == d):
            y = X[:,0]
            theta = np.array([1,0,0,0,0])
        elif (d2 == d):
            y = X[:,1]
            theta = np.array([0,1,0,0,0])
        elif (d3 == d):
            y = X[:,2]
            theta = np.array([0,0,1,0,0])
        elif (d4 == d):
            y = X[:,3]
            theta = np.array([0,0,0,1,0])
        elif (d5 == d):
            y = X[:,4]
            theta = np.array([0,0,0,0,1])
        elif (d6 == d):
            y = y6
            theta = np.zeros(5)
            theta[[0,1]] = theta6
        elif (d7 == d):
            y = y7
            theta = np.zeros(5)
            theta[[0,2]] = theta7
        elif (d8 == d):
            y = y8
            theta = np.zeros(5)
            theta[[0,3]] = theta8
        elif (d9 == d):
            y = y9
            theta = np.zeros(5)
            theta[[0,4]] = theta9
        elif (d10 == d):
            y = y10
            theta = np.zeros(5)
            theta[[1,2]] = theta10
        elif (d11 == d):
            y = y11
            theta = np.zeros(5)
            theta[[1,3]] = theta11
        elif (d12 == d):
            y = y12
            theta = np.zeros(5)
            theta[[1,4]] = theta12
        elif (d13 == d):
            y = y13
            theta = np.zeros(5)
            theta[[2,3]] = theta13
        elif (d14 == d):
            y = y14
            theta = np.zeros(5)
            theta[[2,4]] = theta14
        elif (d15 == d):
            y = y15
            theta = np.zeros(5)
            theta[[3,4]] = theta15
        elif (d16 == d):
            y = y16
            theta = np.zeros(5)
            theta[[0,1,2]] = theta16
        elif (d17 == d):
            y = y17
            theta = np.zeros(5)
            theta[[0,1,3]] = theta17
        elif (d18 == d):
            y = y18
            theta = np.zeros(5)
            theta[[0,1,4]] = theta18
        elif (d19 == d):
            y = y19
            theta = np.zeros(5)
            theta[[0,2,3]] = theta19
        elif (d20 == d):
            y = y20
            theta = np.zeros(5)
            theta[[0,2,4]] = theta20
        elif (d21 == d):
            y = y21
            theta = np.zeros(5)
            theta[[0,3,4]] = theta21
        elif (d22 == d):
            y = y22
            theta = np.zeros(5)
            theta[[1,2,3]] = theta22
        elif (d23 == d):
            y = y23
            theta = np.zeros(5)
            theta[[1,2,4]] = theta23
        elif (d24 == d):
            y = y24
            theta = np.zeros(5)
            theta[[1,3,4]] = theta24
        elif (d25 == d):
            y = y25
            theta = np.zeros(5)
            theta[[2,3,4]] = theta25
        elif (d26 == d):
            y = y26
            theta = np.zeros(5)
            theta[[0,1,2,3]] = theta26
        elif (d27 == d):
            y = y27
            theta = np.zeros(5)
            theta[[0,1,2,4]] = theta27
        elif (d28 == d):
            y = y28
            theta = np.zeros(5)
            theta[[0,1,3,4]] = theta28
        elif (d29 == d):
            y = y29
            theta = np.zeros(5)
            theta[[0,2,3,4]] = theta29
        else:
            y = y30
            theta = np.zeros(5)
            theta[[1,2,3,4]] = theta30
    return[theta,y]

            

def nnls( y, X, niter ):

# Solves min |y-X\theta| st \theta>=0, \sum\theta = 1, using projected gradient. Maximum number of iterations is specified by 'niter'.


    m = X.shape[0]
    p = X.shape[1]
    Xt = X.transpose()
    Sig = np.dot(X.transpose(),X)/m
    SS = Sig
    for i in range(0,10):
        SS = np.dot(Sig,SS)
    L = np.power(np.trace(SS)/p,0.1)
    theta = np.ones(p)/p
    it = 0
    converged = 0
    while ((converged==0)and(it<niter)):
        res = y-np.dot(X,theta)
        grad = -np.dot(Xt,res)/m
        thetanew = project_simplex(theta-grad/L)
        dist = np.linalg.norm(theta-thetanew)
        theta = thetanew
        if (dist<0.00001*np.linalg.norm(theta)):
            converged = 1
        it += 1
    return theta

def nnls_nesterov ( y, X, niter ):

# Solves min |y-X\theta| st \theta>=0, \sum\theta = 1, using 'Nesterov' accelerated projected gradient. Maximum number of iterations is specified by 'niter'.


    m = X.shape[0]
    p = X.shape[1]
    Xt = X.transpose()
    _,s,_ = np.linalg.svd(X)
    smin = np.power(min(s),2)
    L = np.power(max(s),2)
    theta = np.ones(p)/p
    mu = np.ones(p)/p
    it = 0
    converged = 0
    g = max(1,smin)
    while ((converged==0)and(it<niter)):
        t = (smin-g + np.sqrt(pow((m-g),2)+4*g*L))/(2*L)
        thetatemp = theta + ((t*g)/(g+smin*t))*(mu-theta)
        res = y-np.dot(X,thetatemp)
        grad = -np.dot(Xt,res)
        thetanew = project_simplex(theta-grad/L)
        dist = np.linalg.norm(theta-thetanew)
        mu = theta + (thetanew - theta)/t
        theta = thetanew
        if (dist<0.00001*np.linalg.norm(theta)):
            converged = 1
        it += 1
        g = pow(t,2)*L
    return theta

def nnls_fista ( y, X, niter ):

# Solves min |y-X\theta| st \theta>=0, \sum\theta = 1, using Fast Iterative Shrinkage Thresholding Algorithm 'FISTA' by Beck and Teboulle. Maximum number of iterations is specified by 'niter'.

    m = X.shape[0]
    p = X.shape[1]
    Xt = X.transpose()
    Sig = np.dot(X.transpose(),X)/m
    SS = Sig.copy()
    for i in range(0,10):
        SS = np.dot(Sig,SS)
    L = np.power(np.trace(SS)/p,0.1)*1
    theta = np.ones(p)/p
    mu = np.ones(p)/p
    it = 0
    converged = 0
    t = 1
    while ((converged==0)and(it<niter)):
        
        res = y-np.dot(X,mu)
        grad = -np.dot(Xt,res)/m
        thetanew = project_simplex(mu-grad/L)
        tnew = (1 + np.sqrt(1+4*np.power(t,2)))/2
        munew = thetanew + ((t-1)/tnew)*(thetanew-theta)
        dist = np.linalg.norm(theta-thetanew)
        theta = thetanew
        mu = munew
        if (dist<0.00001*np.linalg.norm(theta)):
            converged = 1
        it += 1
    return theta


def check_if_optimal(X,x,threshold=1e-8):

# checks whether 'x' approximates the projection of the origin onto the convex hull of the rows of matrix 'X'. The approximation acceptance threshold is determined by 'threshold'. 

    isoptimal = 1
    n = X.shape[0]
    min_res = 0
    min_ind = -1
    for i in range(0,n):
        res = np.dot(X[i,:]-x, np.transpose(x))
        if (res < min_res):
            min_res = res
            min_ind = i
    if (min_res < -threshold):
        isoptimal = 0
    return [isoptimal, min_ind, min_res]



def gjk_proj(X,m,epsilon=1e-3,threshold=1e-8,niter=10000,verbose=False,method = 'fista', fixed_max_size=float("inf")):

    
# Projects origin onto the convex hull of the rows of 'X' using GJK method with initial simplex size equal to 'm'. The algorithm is by Gilbert, Johnson and Keerthi in their paper 'A fast procedure for computing the distance between complex objects in three-dimensional space'. The input parameters are as below:
# 'epsilon': This is an algorithm parameter determining the threshold that entries of weight vectors that are below 'epsilon' are set to zero. Default = 1e-3.
# 'threshold': The parameter determining the approximation acceptance threshold. Default = 1e-8.
# 'niter': Maximum number of iterations. Default = 10000.
# 'verbose': If set to be True, the algorithm prints out the current set of weights, active set, current estimate of the projection after each iteration. Default = False.
# 'method': If the size of the current active set is larger than 5, this method is used to calculate the projection onto the face created by active set. Options are 'proj_grad' for projected gradient, 'nesterov' for Nesterov accelerated gradient method, 'fista' for FISTA. Default = 'fista'.
# 'fixed_max_size': maximum size of the active set. Default = Inf.

    n = X.shape[0]
    d = X.shape[1]
    m = min(n,m)
    s_ind = np.random.permutation(n)
    s_ind = s_ind[0:m]
    isoptimal = 0
    iter = 0
    weights = np.zeros(n)
    while (isoptimal == 0):
        iter = iter+1
        X_s = X[s_ind,:]
        if (len(s_ind)==2):
            theta, y = proj_line_seg(np.transpose(X_s), np.zeros(d))
        elif (len(s_ind) == 3):
            theta, y = proj_triangle(np.transpose(X_s), np.zeros(d))
        elif (len(s_ind) == 4):
            theta, y = proj_tetrahedron(np.transpose(X_s), np.zeros(d))
        elif (len(s_ind) == 5):
            theta, y = proj_5cell(np.transpose(X_s), np.zeros(d))
        elif (method=='nesterov'):
            theta = nnls_nesterov(np.zeros(d),np.transpose(X_s),niter) 
            y = np.dot(np.transpose(X_s),theta)
        elif (method=='fista'):
            theta = nnls_fista(np.zeros(d),np.transpose(X_s),niter) 
            y = np.dot(np.transpose(X_s),theta)
        else:
            theta = nnls(np.zeros(d),np.transpose(X_s),niter) 
            y = np.dot(np.transpose(X_s),theta)
        weights[s_ind] = theta
        [isoptimal, min_ind, min_res] = check_if_optimal(X, np.transpose(y),threshold=threshold)
        
        ref_ind = (theta>epsilon)
        pruned_ind = np.argmin(theta)
        prune = False
        if (sum(ref_ind)>=fixed_max_size):
            prune = True
                
        
        if (min_ind >= 0):
            if (min_ind in s_ind):
                isoptimal = 1
            else: 
                s_ind = s_ind[ref_ind]
                s_ind = np.append(s_ind, min_ind)
                if prune==True:
                    s_ind = np.delete(s_ind, pruned_ind)
                    prune = False

        if (verbose==True):
            print('X_s=')
            print(X_s)
            print('theta=')
            print(theta)
            print('y=')
            print(y)
            print('ref_ind=')
            print(ref_ind)
            print('s_ind=')
            print(s_ind)
        
        
    return [y, weights]

def wolfe_proj(X,epsilon=1e-6,threshold=1e-8,niter=10000,verbose=False):

# Projects origin onto the convex hull of the rows of 'X' using Wolfe method. The algorithm is by Wolfe in his paper 'Finding the nearest point in A polytope'. The input parameters are as below:
# 'epsilon', 'threshold': Algorithm parameters determining approximation acceptance thresholds. These parameters are denoted as (Z2,Z3) and Z1, in the main paper, respectively. Default values = 1e-6, 1e-8.
# 'niter': Maximum number of iterations. Default = 10000.
# 'verbose': If set to be True, the algorithm prints out the current set of weights, active set, current estimate of the projection after each iteration. Default = False.

    n = X.shape[0]
    d = X.shape[1]
    max_norms = np.min(np.sum(np.abs(X)**2,axis=-1)**(1./2))
    s_ind = np.array([np.argmin(np.sum(np.abs(X)**2,axis=-1)**(1./2))])
    w = np.array([1.0])
    E = np.array([[-max_norms**2, 1.0], [1.0, 0.0]])
    isoptimal = 0
    iter = 0
    while (isoptimal == 0) and (iter <= niter):
        isoptimal_aff = 0
        iter = iter+1
        P = np.dot(w,np.reshape(X[s_ind,:], (len(s_ind), d)))
        new_ind = np.argmin(np.dot(P,X.T))
        max_norms = max(max_norms, np.sum(np.abs(X[new_ind,:])**2))
        if (np.dot(P, X[new_ind,:]) > np.dot(P,P) - threshold*max_norms):
            isoptimal = 1
        elif (np.any(s_ind == new_ind)):
            isoptimal = 1
        else:
            y = np.append(1,np.dot(X[s_ind,:], X[new_ind,:]))
            Y = np.dot(E, y)
            t = np.dot(X[new_ind,:], X[new_ind,:]) - np.dot(y, np.dot(E, y))
            s_ind = np.append(s_ind, new_ind)
            w = np.append(w, 0.0)
            E = np.block([[E + np.outer(Y, Y)/(t+0.0), -np.reshape(Y/(t+0.0), (len(Y),1))], [-Y/(t+0.0), 1.0/(t+0.0)]])
            while (isoptimal_aff == 0):
                v = np.dot(E,np.block([1, np.zeros(len(s_ind))]))
                v = v[1:len(v)]          
                if (np.all(v>epsilon)):
                    w = v
                    isoptimal_aff = 1
                else:
                    POS = np.where((w-v)>epsilon)[0]
                    if (POS.size==0):
                        theta = 1
                    else:
                        fracs = (w+0.0)/(w-v)
                        theta = min(1, np.min(fracs[POS]))
                    w = theta*v + (1-theta)*w
                    w[w<epsilon] = 0
                    if np.any(w==0):
                        remov_ind = np.where(w==0)[0][0]
                        w = np.delete(w, remov_ind)
                        s_ind = np.delete(s_ind, remov_ind)
                        col = E[:, remov_ind+1]
                        E = E - (np.outer(col,col)+0.0)/col[remov_ind+1]
                        E = np.delete(np.delete(E, remov_ind+1, axis=0), remov_ind+1, axis=1)
        
        y = np.dot(X[s_ind,:].T, w)
        if (verbose==True):
            print('X_s=')
            print(X[s_ind,:])
            print('w=')
            print(w)
            print('y=')
            print(y)
            print('s_ind=')
            print(s_ind)

        weights = np.zeros(n)
        weights[s_ind] = w
    return [y, weights]


def palm_nmf_update(H, W, X, l, proj_method='wolfe', m=5, c1=1.2, c2=1.2, proj_low_dim = False, eps_step=1e-4, epsilon='None', threshold=1e-8, niter=10000, method = 'fista', weights_exact = False, fixed_max_size=float("inf")):

# Performs an iteration of PALM algorithm. The inputs are as below.
# 'H': Current matrix of Archetypes.
# 'W': Current matrix of Weights.
# 'X': Input Data points.
# 'l': parameter \lambda of the algorithm.
# 'proj_method': method used for computing the projection onto the convex hull. Options are: 'wolfe' for Wolfe method, 'gjk' for GJK algorithm, 'proj_grad' for projected gradient, 'nesterov' for Nesterov accelerated gradient method, 'fista' for FISTA. Default is 'wolfe'.
# 'm': Original size of the active set used for projection. Used only when GJK method is used for projection. Default is m=5.
# 'c1', 'c2': Parameters for determining the step size of the update. default values are 1.2.
# 'proj_low_dim': If set to be True, the algorithm replaces the data points with their projections onto the principal r-dimensional subspace formed by them. Default is False.
# 'eps_step': Small constant to make sure that the step size of the iteration remains bounded and the PALM iterations remain well-defined. Default value is 1e-4.
# 'epsilon': Plays the role of 'epsilon' in 'wolfe_proj' and 'gjk_proj' functions. Only used when GJK or Wolfe methods used for projection. Default value is equal to their corresponding default value for each GJK or Wolfe method.
# 'threshold': Plays the role of 'threshold' in 'wolfe_proj' and 'gjk_proj' functions. Only used when GJK or Wolfe methods used for projection. Default value is 1-e8.
# 'niter': Maximum number of iterations for computing the projection. Default is 10000.
# 'method': The same as 'method' in 'gjk_proj' function. Only used when GJK method is chosen.
# 'weights_exact': Updates the weights with their 'exact' estimates resulting from solving the constrained non-negative least squares problem after updating 'H' at each iteration. Must be set to False to follow the PALM iterations. 
# 'fixed_max_size': The same as 'fixed_max_size' in 'gjk_proj' function. Only used when GJK method is chosen.

    if (epsilon == 'None') and (proj_method=='wolfe'):
        epsilon = 1e-6
    elif (epsilon == 'None') and (proj_method=='gjk'):
        epsilon = 1e-3
    n = W.shape[0]
    r = W.shape[1]
    d = H.shape[1]
    Hnew = H.copy()
    Wnew = W.copy()
    gamma1 = c1*np.linalg.norm(np.dot(np.transpose(W), W))
    gamma2 = c2*np.linalg.norm(np.dot(H, np.transpose(H)))
    gamma2 = max(gamma2, eps_step)
    res = np.dot(W, H) - X[:]
    H_temp = H.copy() - np.dot(np.transpose(W), res)/gamma1
    for i in range (0,r):
        if (proj_low_dim == True):
            proj_X = project_principal(X,min(d,r))
            if (proj_method=='wolfe'):
                H_grad,_ = wolfe_proj(proj_X-H_temp[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                H_grad,_ = gjk_proj(proj_X-H_temp[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                H_grad = nnls(H_temp[i,:], proj_X.T, niter=niter)
                H_grad = np.dot(proj_X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='nesterov'):
                H_grad = nnls_nesterov(H_temp[i,:], proj_X.T, niter=niter)
                H_grad = np.dot(proj_X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='fista'):
                H_grad = nnls_fista(H_temp[i,:], proj_X.T, niter=niter)
                H_grad = np.dot(proj_X.T, H_grad) - H_temp[i,:]

        else:
            if (proj_method=='wolfe'):
                H_grad,_ = wolfe_proj(X-H_temp[i,:], epsilon=epsilon,threshold=threshold,niter=niter) 
            elif (proj_method=='gjk'):
                H_grad,_ = gjk_proj(X-H_temp[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                H_grad = nnls(H_temp[i,:], X.T, niter=niter)
                H_grad = np.dot(X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='nesterov'):
                H_grad = nnls_nesterov(H_temp[i,:], X.T, niter=niter)
                H_grad = np.dot(X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='fista'):
                H_grad = nnls_fista(H_temp[i,:], X.T, niter=niter)
                H_grad = np.dot(X.T, H_grad) - H_temp[i,:]

        Hnew[i,:] = H_temp[i,:] + (l/(l+gamma1))*H_grad
    
    res = np.dot(W, Hnew) - X
    if weights_exact==False:
        W_temp = W[:] - (1/gamma2)*np.dot(res, np.transpose(Hnew))
        for i in range (0,n):
            Wnew[i,:] = project_simplex(W_temp[i,:])
    else:
        for i in range (0,n):
             if (proj_low_dim==True):
                 if (proj_method == 'wolfe'):
                     _,Wnew[i,:] = wolfe_proj(Hnew - proj_X[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
                 elif (proj_method == 'gjk'):
                     _,Wnew[i,:] = gjk_proj(Hnew - proj_X[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
                 elif (proj_method == 'proj_grad'):
                     Wnew[i,:] = nnls(proj_X[i,:], Hnew.T, niter=niter)
                 elif (proj_method == 'nesterov'):
                     Wnew[i,:] = nnls_nesterov(proj_X[i,:], Hnew.T, niter=niter)
                 elif (proj_method == 'fista'):
                     Wnew[i,:] = nnls_fista(proj_X[i,:], Hnew.T, niter=niter)
             else:
                 if (proj_method == 'wolfe'):
                     _,Wnew[i,:] = wolfe_proj(Hnew - X[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
                 elif (proj_method == 'gjk'):
                     _,Wnew[i,:] = gjk_proj(Hnew - X[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
                 elif (proj_method == 'proj_grad'):
                     Wnew[i,:] = nnls(X[i,:], Hnew.T, niter=niter)
                 elif (proj_method == 'nesterov'):
                     Wnew[i,:] = nnls_nesterov(X[i,:], Hnew.T, niter=niter)
                 elif (proj_method == 'fista'):
                     Wnew[i,:] = nnls_fista(X[i,:], Hnew.T, niter=niter)
        

    return [Wnew, Hnew]

def costfun(W, H, X, l, proj_method='wolfe', m=3, epsilon='None',threshold=1e-8,niter=1000, method = 'fista', fixed_max_size=float("inf")):

# Computes the value of the cost function minimized by PALM iterations. The inputs are as below:
# 'W': Matrix of weights.
# 'H': Matrix of Archetypes.
# 'X': Data matrix.
# 'l': \lambda.
# 'proj_method': The same as in 'palm_nmf_update' function. 
# 'm': The same as in 'palm_nmf_update' function.
# 'epsilon': The same as in 'palm_nmf_update' function.
# 'threshold': The same as in 'palm_nmf_update' function.
# 'niter': The same as in 'palm_nmf_update' function.
# 'method': The same as in 'palm_nmf_update' function.
# 'fixed_max_size': The same as in 'palm_nmf_update' function.

    if (epsilon == 'None') and (proj_method=='wolfe'):
        epsilon = 1e-6
    elif (epsilon == 'None') and (proj_method=='gjk'):
        epsilon = 1e-3
    n = W.shape[0]
    r = W.shape[1]
    d = H.shape[1]
    fH = np.power(np.linalg.norm(X - np.dot(W,H)),2)
    for i in range (0,r):
        if (proj_method == 'wolfe'):
            projHi,_ = wolfe_proj(X-H[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
        elif (proj_method == 'gjk'):
            projHi,_ = gjk_proj(X-H[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
        elif (proj_method=='proj_grad'):
                projHi = nnls(H[i,:], X.T, niter=niter)
                projHi = np.dot(X.T, projHi) - H[i,:]
        elif (proj_method=='nesterov'):
                projHi = nnls_nesterov(H[i,:], X.T, niter=niter)
                projHi = np.dot(X.T, projHi) - H[i,:]
        elif (proj_method=='fista'):
                projHi = nnls_fista(H[i,:], X.T, niter=niter)
                projHi = np.dot(X.T, projHi) - H[i,:]

        fH = fH + l*(np.power(np.linalg.norm(projHi),2))
    
    return fH
    

def palm_nmf(X,r,l = None,lmax = 10, lmin = 0.001, lambda_no=20, c_lambda=1.2, proj_method='wolfe', m=5,H_init= None, W_init = None, maxiter=200, delta=1e-6, c1=1.1, c2=1.1 ,proj_low_dim = False, eps_step=1e-4, epsilon='None', threshold=1e-8, niter=10000, verbose=False, plotit=False, plotloss= True, ploterror= True, oracle=True, H0 = [], weights_exact = False, method = 'fista', fixed_max_size=float("inf")):

# The main function which minimizes the proposed cost function using PALM iterations and outputs the estimates for archetypes, weights, fitting error and estimation error for the archetypes (if the ground truth is known). The inputs are as below:

# 'X': Input data matrix.
# 'r': Rank of the fitted model.
# 'l': \lambda, if not given data driven method is used to find it.
# 'lmax': maximum of the search range for \lambda. Default value is 10.
# 'lmin': minimum of the search range for \lambda. Default value is 0.001.
# 'lambda_no': number of \lambdas in the range [lmin, lmax] used for search in finding appropriate \lambda. Default is 20.
# 'c_lambda': constant 'c' used in the data driven method for finding \lambda. Default is 1.2.
# 'proj_method': The same as in 'palm_nmf_update' function.
# 'm': The same as in 'palm_nmf_update' function.
# 'H_init': Initial value for the archetype matrix H. If not given, successive projection method is used to find an initial point.
# 'W_init': Initial value for the weights matrix H. If not given, successive projection method is used to find an initial point.
# 'maxiter': Maximum number of iterations of PALM algorithm. Default value is 200.
# 'delta': PALM Iterations are terminated when the frobenius norm of the differences between W,H estimates for successive iterations are less than 'delta'. Default value is 1e-6.
# 'c1': The same as in 'palm_nmf_update' function. Default value is 1.1.
# 'c2': The same as in 'palm_nmf_update' function. Default value is 1.1.
# 'proj_low_dim': The same as in 'palm_nmf_update' function.
# 'eps_step': The same as in 'palm_nmf_update' function.
# 'epsilon': The same as in 'palm_nmf_update' function.
# 'threshold': The same as in 'palm_nmf_update' function.
# 'niter': The same as in 'palm_nmf_update' function.
# 'verbose': If it is 'True' the number of taken iterations is given. If the ground truth is known, the Loss is also typed after each iteration. Default value is False.
# 'plotit': For the case that data points are in 2 dimensions, if 'plotit' is true, data points and estimate for archetypes are plotted after each iteration. Default value is False.
# 'plotloss': If it is True and the ground truth is known, the Loss in estimating archetypes versus iteration is plotted. Default value is True.
# 'ploterror': If it is True the minimized cost function versus iteration is plotted. Default value is True.
# 'oracle': If it is True then the ground truth archetypes are given in H0. The Default value is True.
# 'H0': Ground truth archetypes. Default is empty array.
# 'weights_exact': The same as in 'palm_nmf_update' function. 
# 'method': The same as in 'palm_nmf_update' function.
# 'fixed_max_size': The same as in 'palm_nmf_update' function.
                                                                                                                                                                           
                                                                                                                                                                                                                     
    if (epsilon == 'None') and (proj_method=='wolfe'):
        epsilon = 1e-6
    elif (epsilon == 'None') and (proj_method=='gjk'):
        epsilon = 1e-3
    if (l == None):
        lambdas = np.geomspace(lmin, lmax, lambda_no)
    else:
        lambdas = np.array([l])
        lambda_no = 1

    n = X.shape[0]
    d = X.shape[1]

    if (d<=r):
        pca_loss = 0
    else:
        proj_X = project_principal(X,r)
        pca_loss = np.linalg.norm(X-proj_X)

    l_no = 0
    l_stop = 0

    while (l_stop == 0):
        
        l = lambdas[l_no]
        print('lambda =')
        print(l)
        Err = np.array([])
        L = np.array([])
        n = X.shape[0]
        d = X.shape[1]
        if (plotit==True):
            plt.ion()
            plot_data(X, 'b')
        if H_init is None:
            H = initH(X,r)
        else:
            H = H_init.copy()
        if W_init is None:
            W = np.ones((n, r))/r
        else:
            W = W_init.copy()
        if (oracle==True):
            L=[np.sqrt(D_distance(H0, H))]
        Err = [np.linalg.norm(np.dot(W,H) - X)]
        converged = 0
        iter = 1
        conv_hull_loss = 0
        while ((iter<=maxiter) and (converged==0)):
            Wnew, Hnew = palm_nmf_update(H, W, X, l=l, proj_method=proj_method, m=m, c1=c1, c2=c2, proj_low_dim = proj_low_dim, eps_step=eps_step, epsilon=epsilon, threshold=threshold, niter=niter, weights_exact = weights_exact, method = method, fixed_max_size=fixed_max_size)
     
            if ((np.linalg.norm(H - Hnew)<=delta)and((np.linalg.norm(W - Wnew)<=delta))):
                converged = 1
            H = Hnew.copy()
            W = Wnew.copy()
            iter = iter + 1
            Err.append(np.linalg.norm(np.dot(W,H) - X))
            if (oracle==True):
                L.append(np.sqrt(D_distance(H0, H)))
                if (verbose==True):
                    print('Loss:')
                    print(L[iter-2])
            if (verbose==True):
                print('iter')
                print(iter)
            if (plotit==True):
                hplt = plot_H(H, 'r','o')
                plt.pause(0.05)
                hplt.remove()

        print('number of iterations:')
        print(iter-1)

        if (oracle==True):
            print('Final Loss in Estimating Archetypes:')
            print(L[iter-2])

        for j in range (0,n):
            if (proj_method=='wolfe'):
                projXj,_ = wolfe_proj(Hnew-X[j,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                projXj,_ = gjk_proj(Hnew-X[j,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                projXj = nnls(X[j,:], Hnew.T, niter=niter)
                projXj = np.dot(Hnew.T, projXj) - X[j,:]
            elif (proj_method=='nesterov'):
                projXj = nnls_nesterov(X[j,:], Hnew.T, niter=niter)
                projXj = np.dot(Hnew.T, projXj) - X[j,:]
            elif (proj_method=='fista'):
                projXj = nnls_fista(X[j,:], Hnew.T, niter=niter)
                projXj = np.dot(Hnew.T, projXj) - X[j,:]
            conv_hull_loss = conv_hull_loss + (np.power(np.linalg.norm(projXj),2))
        
        conv_hull_loss = np.sqrt(conv_hull_loss)

        l_lambda = conv_hull_loss - pca_loss
        if (l_no == 0):
            l_lambda0 = l_lambda
        if (l_no == lambda_no-1) or (l_lambda >= l_lambda0*c_lambda):
            l_stop = 1

        
        if (plotloss == True) and (l_stop==1):   
            figlossnmf = plt.figure()
            plt.plot(L)
            plt.yscale('log')
            figlossnmf.suptitle('Loss vs iteration', fontsize=20)
            plt.xlabel('Iteration', fontsize=18)
            plt.ylabel('Loss', fontsize=16)
            plt.show()
        if (ploterror == True) and (l_stop==1):
            figerrornmf = plt.figure()
            plt.plot(Err)
            plt.yscale('log')
            figerrornmf.suptitle('Reconstruction Error vs iteration', fontsize=20)
            plt.xlabel('Iteration', fontsize=18)
            plt.ylabel('Error', fontsize=16)
            plt.show()
        l_no = l_no + 1


    return [Wnew, Hnew, L, Err]

def acc_palm_nmf_update(H, H_old, I, W, W_old, Y, t, t_old, X, l, proj_method='wolfe', m=5, c1=1.2, c2=1.2, proj_low_dim = False, eps_step=1e-4, epsilon='None', threshold=1e-8, niter=10000, method = 'fista', weights_exact = False, fixed_max_size=float("inf")):

# Performs one iteration of the Accelerated PALM iteration. The parameters have similar meanings as in 'palm_nmf_update' function.

    if (epsilon == 'None') and (proj_method=='wolfe'):
        epsilon = 1e-6
    elif (epsilon == 'None') and (proj_method=='gjk'):
        epsilon = 1e-3    
    n = W.shape[0]
    r = W.shape[1]
    d = H.shape[1]
    Hnew = H.copy()
    Wnew = W.copy()
    Inew = I.copy()
    Jnew = I.copy()
    Ynew = Y.copy()
    Znew = Y.copy()
    gamma1 = c1*np.linalg.norm(np.dot(np.transpose(W), W))
    gamma2 = c2*np.linalg.norm(np.dot(H, np.transpose(H)))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    gamma2 = max(gamma2, eps_step)
    
    G = H + (t_old/t)*(I - H) + ((t_old-1)/t)*(H - H_old)
    V = W + (t_old/t)*(Y - W) + ((t_old-1)/t)*(W - W_old)
    
    res = np.dot(V, G) - X[:]
    G_temp = G.copy() - np.dot(np.transpose(V), res)/gamma1

    for i in range (0,r):
        if (proj_low_dim == True):
            proj_X = project_principal(X,min(d,r))
            if (proj_method=='wolfe'):
                G_grad,_ = wolfe_proj(proj_X-G_temp[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                G_grad,_ = gjk_proj(proj_X-G_temp[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                G_grad = nnls(G_temp[i,:], Proj_X.T, niter=niter)
                G_grad = np.dot(Proj_X.T,G_grad) - G_temp[i,:]
            elif (proj_method=='nesterov'):
                G_grad = nnls_nesterov(G_temp[i,:], Proj_X.T, niter=niter)
                G_grad = np.dot(Proj_X.T,G_grad) - G_temp[i,:]
            elif (proj_method=='fista'):
                G_grad = nnls_fista(G_temp[i,:], Proj_X.T, niter=niter)
                G_grad = np.dot(Proj_X.T,G_grad) - G_temp[i,:]

        else:
            if (proj_method=='wolfe'):
                G_grad,_ = wolfe_proj(X-G_temp[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                G_grad,_ = gjk_proj(X-G_temp[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                G_grad = nnls(G_temp[i,:], X.T, niter=niter)
                G_grad = np.dot(X.T,G_grad) - G_temp[i,:]
            elif (proj_method=='nesterov'):
                G_grad = nnls_nesterov(G_temp[i,:], X.T, niter=niter)
                G_grad = np.dot(X.T,G_grad) - G_temp[i,:]
            elif (proj_method=='fista'):
                G_grad = nnls_fista(G_temp[i,:], X.T, niter=niter)
                G_grad = np.dot(X.T,G_grad) - G_temp[i,:]
        Inew[i,:] = G_temp[i,:] + (l/(l+gamma1))*G_grad

    res = np.dot(W, H) - X[:]
    H_temp = H.copy() - np.dot(np.transpose(W), res)/gamma1

    for i in range (0,r):
        if (proj_low_dim == True):
            proj_X = project_principal(X,min(d,r))
            if (proj_method=='wolfe'):
                H_grad,_ = wolfe_proj(proj_X-H_temp[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                H_grad,_ = gjk_proj(proj_X-H_temp[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                H_grad = nnls(H_temp[i,:], proj_X.T, niter=niter)
                H_grad = np.dot(Proj_X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='nesterov'):
                H_grad = nnls_nesterov(H_temp[i,:], proj_X.T, niter=niter)
                H_grad = np.dot(Proj_X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='fista'):
                H_grad = nnls_fista(H_temp[i,:], proj_X.T, niter=niter)
                H_grad = np.dot(Proj_X.T, H_grad) - H_temp[i,:]
            

        else:
            if (proj_method=='wolfe'):
                H_grad,_ = wolfe_proj(X-H_temp[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                H_grad,_ = gjk_proj(X-H_temp[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                H_grad = nnls(H_temp[i,:], X.T, niter=niter)
                H_grad = np.dot(X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='nesterov'):
                H_grad = nnls_nesterov(H_temp[i,:], X.T, niter=niter)
                H_grad = np.dot(X.T, H_grad) - H_temp[i,:]
            elif (proj_method=='fista'):
                H_grad = nnls_fista(H_temp[i,:], X.T, niter=niter)
                H_grad = np.dot(X.T, H_grad) - H_temp[i,:]
        Jnew[i,:] = H_temp[i,:] + (l/(l+gamma1))*H_grad
    
    res = np.dot(V, Inew) - X
    if weights_exact==False:
        V_temp = V[:] - (1/gamma2)*np.dot(res, np.transpose(Inew))
        for i in range (0,n):
            Ynew[i,:] = project_simplex(V_temp[i,:])
    else:
        for i in range (0,n):
             if (proj_low_dim==True):
                 if (proj_method =='wolfe'):
                     _,Ynew[i,:] = wolfe_proj(Inew - proj_X[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
                 elif (proj_method=='gjk'):
                     _,Ynew[i,:] = gjk_proj(Inew - proj_X[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
                 elif (proj_method=='proj_grad'):
                     Ynew[i,:] = nnls(proj_X[i,:], Inew, niter=niter)
                 elif (proj_method=='nesterov'):
                     Ynew[i,:] = nnls_nesterov(proj_X[i,:], Inew, niter=niter)
                 elif (proj_method=='fista'):
                     Ynew[i,:] = nnls_fista(proj_X[i,:], Inew, niter=niter)
    
             else:
                 if (proj_method =='wolfe'):
                     _,Ynew[i,:] = wolfe_proj(Inew - X[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
                 elif (proj_method=='gjk'):
                     _,Ynew[i,:] = gjk_proj(Inew - X[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
                 elif (proj_method=='proj_grad'):
                     Ynew[i,:] = nnls(X[i,:], Inew, niter=niter)
                 elif (proj_method=='nesterov'):
                     Ynew[i,:] = nnls_nesterov(X[i,:], Inew, niter=niter)
                 elif (proj_method=='fista'):
                     Ynew[i,:] = nnls_fista(X[i,:], Inew, niter=niter)

    res = np.dot(W, Jnew) - X
    if weights_exact==False:
        W_temp = W[:] - (1/gamma2)*np.dot(res, np.transpose(Inew))
        for i in range (0,n):
            Znew[i,:] = project_simplex(W_temp[i,:])
    else:
        for i in range (0,n):
            if (proj_low_dim==True):
                if (proj_method =='wolfe'):
                    _,Znew[i,:] = wolfe_proj(Jnew - proj_X[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
                elif (proj_method=='gjk'):
                     _,Znew[i,:] = gjk_proj(Jnew - proj_X[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
                elif (proj_method =='proj_grad'):
                    Znew[i,:] = nnls(proj_X[i,:], Jnew, niter=niter)
                elif (proj_method =='nesterov'):
                    Znew[i,:] = nnls_nesterov(proj_X[i,:], Jnew, niter=niter)
                elif (proj_method =='fista'):
                    Znew[i,:] = nnls_fista(proj_X[i,:], Jnew, niter=niter)
    
            else:
                if (proj_method =='wolfe'):
                    _,Znew[i,:] = wolfe_proj(Jnew - X[i,:], epsilon=epsilon,threshold=threshold,niter=niter)
                elif (proj_method=='gjk'):
                    _,Znew[i,:] = gjk_proj(Jnew - X[i,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
                elif (proj_method =='proj_grad'):
                    Znew[i,:] = nnls(X[i,:], Jnew, niter=niter)
                elif (proj_method =='nesterov'):
                    Znew[i,:] = nnls_nesterov(X[i,:], Jnew, niter=niter)
                elif (proj_method =='fista'):
                    Znew[i,:] = nnls_fista(X[i,:], Jnew, niter=niter)


    tnew = (np.sqrt(4*np.power(t,2) + 1) + 1)/2
    
    if (proj_low_dim==True):
        cost1 = costfun(Ynew, Inew, proj_X, l,  proj_method=proj_method, m=m, epsilon=epsilon, threshold=threshold, niter=1000, method = method, fixed_max_size=fixed_max_size)
        cost2 = costfun(Znew, Jnew, proj_X, l,  proj_method=proj_method, m=m, epsilon=epsilon, threshold=threshold, niter=1000, method = method, fixed_max_size=fixed_max_size)
    else:
        cost1 = costfun(Ynew, Inew, X, l,  proj_method=proj_method, m=m, epsilon=epsilon, threshold=threshold, niter=1000, method = method, fixed_max_size=fixed_max_size)
        cost2 = costfun(Znew, Jnew, X, l,  proj_method=proj_method, m=m, epsilon=epsilon, threshold=threshold, niter=1000, method = method, fixed_max_size=fixed_max_size)

    if (cost1<=cost2):
        Wnew = Ynew
        Hnew = Inew
    else:
        Wnew = Znew
        Hnew = Jnew

    return [Hnew, Inew, Wnew, Ynew, tnew]


def acc_palm_nmf(X,r,l=None, lmax = 10, lmin = 0.001, lambda_no = 20, c_lambda=1.2, proj_method='wolfe', m=5,H_init= None, W_init = None, maxiter=200, delta=1e-6, c1=1.1, c2=1.1 ,proj_low_dim = False, eps_step=1e-4, epsilon='None', threshold=1e-8, niter=10000, verbose=False, plotit=False, plotloss= True, ploterror= True, oracle=True, H0 = [], weights_exact = False, method = 'fista', fixed_max_size=float("inf")):

# The main function which minimizes the proposed cost function using accelerated PALM iterations. The parameters have similar meanings as in 'palm_nmf' function.


    if (epsilon == 'None') and (proj_method=='wolfe'):
        epsilon = 1e-6
    elif (epsilon == 'None') and (proj_method=='gjk'):
        epsilon = 1e-3
    if (l == None):
        lambdas = np.geomspace(lmin, lmax, lambda_no)
    else:
        lambdas = np.array([l])
        lambda_no = 1

    n = X.shape[0]
    d = X.shape[1]

    if (d<=r):
        pca_loss = 0
    else:
        proj_X = project_principal(X,r)
        pca_loss = np.linalg.norm(X-proj_X)

    l_no = 0
    l_stop = 0

    while (l_stop == 0):
        l = lambdas[l_no]
        print('lambda =')
        print(l)
        Err = np.array([])
        L = np.array([])
        n = X.shape[0]
        d = X.shape[1]
        if (plotit==True):
            plt.ion()
            plot_data(X, 'b')
        if H_init is None:
            H = initH(X,r)
        else:
            H = H_init.copy()
        if W_init is None:
            W = np.ones((n, r))/r
        else:
            W = W_init.copy()
        if (oracle==True):
            L=[np.sqrt(D_distance(H0, H))]
        Err = [np.linalg.norm(np.dot(W,H) - X)]
        H_old = H.copy()
        W_old = W.copy()
        I = H.copy()
        Y = W.copy()
        t = 1.0
        t_old = 0.0
        converged = 0
        iter = 1
        conv_hull_loss = 0
        while ((iter<=maxiter) and (converged==0)):
            Hnew, Inew, Wnew, Ynew, tnew = acc_palm_nmf_update(H, H_old, I, W, W_old, Y, t, t_old, X, l=l,  proj_method=proj_method,  m=m, c1=c1, c2=c2, proj_low_dim = proj_low_dim, eps_step=eps_step, epsilon=epsilon, threshold=threshold, niter=niter, weights_exact = weights_exact, method = method, fixed_max_size=fixed_max_size)
     
            if ((np.linalg.norm(H - Hnew)<=delta)and((np.linalg.norm(W - Wnew)<=delta))):
                converged = 1
        
            H_old = H.copy()
            W_old = W.copy()
            t_old = t
            t = tnew
            I = Inew
            Y = Ynew
            H = Hnew.copy()
            W = Wnew.copy()
            iter = iter + 1
            Err.append(np.linalg.norm(np.dot(W,H) - X))
            if (oracle==True):
                L.append(np.sqrt(D_distance(H0, H)))
                if (verbose==True):
                    print('Loss:')
                    print(L[iter-2])
            if (verbose==True):
                print('iter')
                print(iter)
            if (plotit==True):
                hplt = plot_H(H, 'r','o')
                plt.pause(0.05)
                hplt.remove()

        print('number of iterations:')
        print(iter-1)

        if (oracle==True):
            print('Final Loss in Estimating Archetypes:')
            print(L[iter-2])

        for j in range (0,n):
            if (proj_method=='wolfe'):
                projXj,_ = wolfe_proj(Hnew-X[j,:], epsilon=epsilon,threshold=threshold,niter=niter)
            elif (proj_method=='gjk'):
                projXj,_ = gjk_proj(Hnew-X[j,:], m=m, epsilon=epsilon,threshold=threshold,niter=niter, method = method, fixed_max_size=fixed_max_size)
            elif (proj_method=='proj_grad'):
                projXj = nnls(X[j,:], Hnew.T, niter=niter)
                projXj = np.dot(Hnew.T, projXj) - X[j,:]
            elif (proj_method=='nesterov'):
                projXj = nnls_nesterov(X[j,:], Hnew.T, niter=niter)
                projXj = np.dot(Hnew.T, projXj) - X[j,:]
            elif (proj_method=='fista'):
                projXj = nnls_fista(X[j,:], Hnew.T, niter=niter)
                projXj = np.dot(Hnew.T, projXj) - X[j,:]
            conv_hull_loss = conv_hull_loss + (np.power(np.linalg.norm(projXj),2))
        
        conv_hull_loss = np.sqrt(conv_hull_loss)
        
        l_lambda = conv_hull_loss - pca_loss
        if (l_no == 0):
            l_lambda0 = l_lambda
        if (l_no == lambda_no-1) or (l_lambda >= l_lambda0*c_lambda):
            l_stop = 1

        if (plotloss == True) and (l_stop==1):   
            figlossnmf = plt.figure()
            plt.plot(L)
            plt.yscale('log')
            figlossnmf.suptitle('Loss vs iteration', fontsize=20)
            plt.xlabel('Iteration', fontsize=18)
            plt.ylabel('Loss', fontsize=16)
            plt.show()
        if (ploterror == True) and (l_stop==1):
            figerrornmf = plt.figure()
            plt.plot(Err)
            plt.yscale('log')
            figerrornmf.suptitle('Reconstruction Error vs iteration', fontsize=20)
            plt.xlabel('Iteration', fontsize=18)
            plt.ylabel('Error', fontsize=16)
            plt.show()
        l_no = l_no + 1

    return [Wnew, Hnew, L, Err]

