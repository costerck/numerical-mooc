
# coding: utf-8

# Coding assignement :
# 
# We have to solve the following partial differential equations :
# 
# \begin{align}
# \frac{\partial u}{\partial t} &= D_u \nabla ^2 u - uv^2 + F(1-u)\\
# \frac{\partial v}{\partial t} &= D_v \nabla ^2 v + uv^2 - (F + k)v
# \end{align}

# Let's discretize our equation :
# 
# \begin{eqnarray}
# \frac{u^{n+1}_{i,j} - u^n_{i,j}}{\Delta t}
# = \frac{Du}{\delta^2} &\left( u^{n+1}_{i+1, j} + u^{n+1}_{i-1,j} - 4u^{n+1}_{i,j} + u^{n+1}_{i, j+1} + u^{n+1}_{i,j-1}\right) - u^{n}_{i, j}(v^{n}_{i, j})^2 + F(1 - u^{n+1}_{i, j})
# \end{eqnarray}
# \begin{eqnarray}
# \frac{v^{n+1}_{i,j} - v^n_{i,j}}{\Delta t}
# = \frac{Dv}{\delta^2} &\left( v^{n+1}_{i+1, j} + v^{n+1}_{i-1,j} - 4v^{n+1}_{i,j} + v^{n+1}_{i, j+1} + v^{n+1}_{i,j-1}\right) + u^{n}_{i, j}(v^{n}_{i, j})^2 - (F + k) v^{n+1}_{i, j}
# \end{eqnarray}
# 
# Wich leads to :
# 
# \begin{equation}
# -u^{n+1}_{i-1,j} - u^{n+1}_{i+1,j} + \left(\frac{\delta^2}{Du \Delta t} + 4 + F\right) u^{n+1}_{i,j} - u^{n+1}_{i,j-1}-u^{n+1}_{i,j+1} = \frac{\delta^2}{Du}\left(\frac{u^n_{i,j}}{\Delta t} + F - u^{n}_{i, j}(v^{n}_{i, j})^2\right)
# \end{equation}
# \begin{equation}
# -v^{n+1}_{i-1,j} - v^{n+1}_{i+1,j} + \left(\frac{\delta^2}{Dv \Delta t} + 4 + F + k\right) v^{n+1}_{i,j} - v^{n+1}_{i,j-1}-v^{n+1}_{i,j+1} = \frac{\delta^2}{Dv} \left( \frac{v^n_{i,j}}{ \Delta t} + u^{n}_{i, j}(v^{n}_{i, j})^2
#  \right)\end{equation}

# In[11]:

#importing libraries
import numpy
from matplotlib import pyplot
import matplotlib.cm as cm
from scipy.linalg import solve


# In[12]:

#setting initial conditions as asked in the assignement
n = 192

Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.065 # Bacteria 1 

dh = 5/(n-1)

T = 8000

dt = .9 * dh**2 / (4*max(Du,Dv))

nt = int(T/dt)


# In[13]:

uvinitial = numpy.load('./data/uvinitial.npz')
U = uvinitial['U']
V = uvinitial['V']


# In[14]:

fig = pyplot.figure(figsize=(8,5))
pyplot.subplot(121)
pyplot.imshow(U, cmap = cm.RdBu)
pyplot.xticks([]), pyplot.yticks([]);
pyplot.subplot(122)
pyplot.imshow(V, cmap = cm.RdBu)
pyplot.xticks([]), pyplot.yticks([]);


# In[15]:


def constructMatrix(n, sigma):
    """ Generate implicit matrix for 2D Gray-Scott model with Neumann BC
    Assumes dx = dy
    
    Parameters:
    ----------
    nx   : int
        number of discretization points in x
    ny   : int
        number of discretization points in y
    sigma: float
        coefficient in the diagonal (i.e. sigma + F for u and sigma + F + k for v)
        
    Returns:
    -------
    A: 2D array of floats
        Matrix of implicit 2D heat equation
    """
    
    A = numpy.zeros(((n-2)*(n-2),(n-2)*(n-2)))
    
    row_number = 0 # row counter
    for j in range(1,n-1):
        for i in range(1,n-1):
            
            # Corners
            if i==1 and j==1: # Bottom left corner
                A[row_number,row_number] = sigma + 2 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number+n-2] = -1   # fetch j+1
                
            elif i==n-2 and j==1: # Bottom right corner
                A[row_number,row_number] = sigma+2 # Set diagonal
                A[row_number,row_number-1] = -1      # Fetch i-1
                A[row_number,row_number+n-2] = -1   # fetch j+1
                
            elif i==1 and j==n-2: # Top left corner
                A[row_number,row_number] = sigma+3   # Set diagonal
                A[row_number,row_number+1] = -1        # fetch i+1
                A[row_number,row_number-(n-2)] = -1   # fetch j-1
                
            elif i==n-2 and j==n-2: # Top right corner
                A[row_number,row_number] = sigma+2   # Set diagonal
                A[row_number,row_number-1] = -1        # Fetch i-1
                A[row_number,row_number-(n-2)] = -1   # fetch j-1
              
            # Sides
            elif i==1: # Left boundary
                A[row_number,row_number] = sigma+3 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number+n-2] = -1   # fetch j+1
                A[row_number,row_number-(n-2)] = -1 # fetch j-1
            
            elif i==n-2: # Right boundary
                A[row_number,row_number] = sigma+3 # Set diagonal
                A[row_number,row_number-1] = -1      # Fetch i-1
                A[row_number,row_number+n-2] = -1   # fetch j+1
                A[row_number,row_number-(n-2)] = -1 # fetch j-1
                
            elif j==1: # Bottom boundary
                A[row_number,row_number] = sigma+3 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number-1] = -1      # fetch i-1
                A[row_number,row_number+n-2] = -1   # fetch j+1
                
            elif j==n-2: # Top boundary
                A[row_number,row_number] = sigma+3 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number-1] = -1      # fetch i-1
                A[row_number,row_number-(n-2)] = -1 # fetch j-1
             
            # Interior points
            else:
                A[row_number,row_number] = sigma+4 # Set diagonal
                A[row_number,row_number+1] = -1      # fetch i+1
                A[row_number,row_number-1] = -1      # fetch i-1
                A[row_number,row_number+n-2] = -1   # fetch j+1
                A[row_number,row_number-(n-2)] = -1 # fetch j-1
                
            row_number += 1 # Jump to next row of the matrix!
    
    return A           


# In[16]:

def generateRHSu(n, sigma, dt, F, v, u):
    """ Generates right-hand side for 2D implicit heat equation with Neumann BCs
        Assumes dx=dy, Neumann BCs = 0
        
        Paramenters:
        -----------
        nx    : int
            number of discretization points in x
        ny    : int
            number of discretization points in y
        sigma : float
            dx^2/Du
        dt    : float
            time-step
        F     : float
            feed rate
        v     : array of float
            Concentration of v in current time
        u     : array of float
            Concentration of u in current time
        
        
        Returns:
        -------
        RHS  : array of float
            Right hand side of 2D implicit Gray-Scott model equation
    """
    RHS = numpy.zeros((n-2)*(n-2))
    
    row_number = 0 # row counter
    for j in range(1,n-1):
        for i in range(1,n-1):
            
                RHS[row_number] = sigma * (u[j,i]/dt - u[j,i]*v[j,i]*v[j,i] + F)
                row_number += 1 #Jump to next row!
    
    return RHS

def generateRHSv(nx, sigma, dt, Fk, v, u):
    """Generates right-hand side for 2D implicit heat equation with Neumann BCs
        Assumes dx=dy, Neumann BCs = 0
        
        Paramenters:
        -----------
        nx    : int
            number of discretization points in x
        ny    : int
            number of discretization points in y
        sigma : float
            dx^2/Dv
        dt    : float
            time-step
        Fk     : float
            feed + kill rate
        v     : array of float
            Concentration of v in current time
        u     : array of float
            Concentration of u in current time
        
        
        Returns:
        -------
        RHS  : array of float
            Right hand side of 2D implicit Gray-Scott model equation
        """
    RHS = numpy.zeros((n-2)*(n-2))
    
    row_number = 0 # row counter
    for j in range(1,n-1):
        for i in range(1,n-1):
                
            RHS[row_number] = sigma * (v[j,i]/dt + u[j,i]*v[j,i]*v[j,i])  
            row_number += 1 #Jump to next row!
            
        return RHS


# In[17]:

def map_1Dto2D(n, u_1D):
    """ Takes solution of linear system, stored in 1D, 
    and puts them in a 2D array with the BCs
    Valid Neumann with zero flux top and right
        
    Parameters:
    ----------
        nx  : int
            number of nodes in x direction
        ny  : int
            number of nodes in y direction
        u_1D: array of floats
            solution of linear system
            
    Returns:
    -------
        T: 2D array of float
            solution stored in 2D array with BCs
    """
    u = numpy.zeros((n,n))
    
    row_number = 0
    for j in range(1,n-1):
        for i in range(1,n-1):
            u[j,i] = u_1D[row_number]
            row_number += 1
    # Neumann BC
    u[0,:] = u[1,:]
    u[:,0] = u[:,1]
    u[-1,:] = u[-2,:]
    u[:,-1] = u[:,-2]
    
    return T   


# In[18]:

def takeAStep(A_u, A_v, u, v, sigmaU, sigmaV, F, k, n, dt):
    """
    
    Parameters:
    -----------
    u : 2D array of float
        concentration of u
    v : 2D array of float
        concentration of v
    sigmaU : float
        Du*dt/dx^2
    sigmaV : float
        Dv*dt/dx^2
    F : float
        Feed rate
    k : float
        kill rate
    """
    
    bu = generateRHSu(n, dh*dh/Du, dt, F, v, u)
    bv = generateRHSv(n, dh*dh/Dv, dt, F+k, v, u)
    u_1D = solve(A_u, bu)
    v_1D = solve(A_v, bv)
    u = map_1Dto2D(n, u_1D)
    v = map_1Dto2D(n, v_1D)
    
    return 0
    


# In[19]:

A_u = constructMatrix(n, dh*dh/Du/dt + F)
A_v = constructMatrix(n, dh*dh/Dv/dt + F + k)


# In[ ]:




# In[20]:

import time
begin = time.time()
for i in range(50):
    if i%10==0:
        print(int(i/10))
    takeAStep(A_u, A_v, U, V, dh*dh/dt/Du, dh*dh/dt/Dv, F, k, n, dt)
print(time.time()-begin)


# In[36]:


pyplot.figure(figsize=(8,5))
pyplot.contourf(x,y,U,20,cmap=cm.viridis)
pyplot.xlabel('$x$')
pyplot.ylabel('$y$')
pyplot.colorbar();


# In[ ]:



