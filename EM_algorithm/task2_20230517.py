import numpy as np
import matplotlib.pyplot as plt

n = 10000
m = 2

np.random.seed(0)
def data_generate(n):
    return (np.random.randn(n) + np.where(np.random.rand(n) > 0.3, 2., -2.))

x= data_generate(n)
fig1 = plt.figure()
plt.hist(x)
fig1.savefig('./histogram')

def Gaussian(x, mu, sigma, j):
    return np.exp(-((x-mu[j])**2)/2/sigma[j]**2)/np.sqrt(2*np.pi*sigma[j]**2)

def calcu_eta(x, mu, sigma, w, m=2):
    sum = 0
    for i in range(m):
        sum += w[i]*Gaussian(x, mu, sigma, i)
    
    eta = np.zeros((m, x.shape[0]))
    for i in range(m):
        eta[i, :] = w[i]*Gaussian(x, mu, sigma, i)/sum

    return eta

def w_update(eta):
    return np.sum(eta, axis=1)/eta.shape[1]

def mu_update(eta, x):
    return np.dot(eta, x)/np.sum(eta, axis=1)

def sigma_update(eta, x, mu, d=1):
    dot = ((np.repeat(x[:,np.newaxis],2,axis=1) - mu)**2).T
    denominator = np.sum(eta*dot, axis=1)
    numerator = d * np.sum(eta, axis=1)
    return np.sqrt(denominator/numerator)

w_list=[]
mu_list = []
sigma_list = []

#初期値
mu = np.array([-3, 4])
sigma = np.array([0.2, 0.5])
w = np.array([0.3, 0.7])

w_list.append(w)
sigma_list.append(sigma)
mu_list.append(mu)

#パラメータの更新
for i in range(10):
    eta = calcu_eta(x, mu, sigma, w, m)
    w = w_update(eta)
    sigma = sigma_update(eta, x, mu)
    mu = mu_update(eta, x)

    w_list.append(w)
    sigma_list.append(sigma)
    mu_list.append(mu)


q_list = []

for j in range(len(mu_list)):
    q = 0
    for i in range(m):  
        q += w_list[j][i]*Gaussian(x, mu_list[j], sigma_list[j], i)
        q_list.append(q)


for k in [0,2,5,10]:
    fig2 = plt.figure()
    plt.plot(x, q_list[k], '.',label=f'{k}EM algorithm')
    plt.legend(loc='upper right')
    fig2.savefig(f'./q_graph[{k}]')