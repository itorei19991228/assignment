import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

np.random.seed(0)
n = 1000
d = 2
S = np.zeros((n,d))
S[:,0] = np.random.rand(n)
S[:,1] = np.random.randn(n)

S[0,1] = 9
S = (S - np.mean(S, axis=0))/np.std(S, axis=0)
M = np.array([[1, 3], [5, 3]])
X = np.dot(S, M.T)
X = np.dot(np.linalg.inv(scipy.linalg.sqrtm(np.cov(X.T))), X.T).T


def g1(s):
    return s**3

def g1_dot(s):
    return 3*s**2

def g2(s):
    return np.tanh(s)

def g2_dot(s):
    return 1 - (np.tanh(s))**2
    
def Newton(A, func, func_dot, id):
    b_vec = np.array([1,0])
    sum1 = 0
    sum2 = np.zeros(2)
    n = A.shape[0]
    for k in range(n):
        sum1 += func_dot(np.dot(b_vec, A[k,:].T))
        sum2 += A[k,:] * func(np.dot(b_vec, A[k,:].T))
    
    b_vec = (b_vec * sum1 - sum2) / n
    b_vec = b_vec / np.linalg.norm(b_vec)
    
    l = 100
    point = np.zeros((2*l,2))
    for i in range(-l,l):
        point[i+l,:] = b_vec  * i * 0.1

    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], 'ro')
    ax.plot(point[:, 0], point[:, 1], color = "blue")
    plt.savefig(f'20230510_task2_{id}')
    

Newton(X, g1, g1_dot, 1)
Newton(X, g2, g2_dot, 2)



