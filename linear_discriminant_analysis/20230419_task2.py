import numpy as np
import matplotlib
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)


def generate_sample(n, alpha):
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1
    mean1, mean2 = np.array([2, 0]), np.array([-2, 0])
    cov = np.array([[1, 0], [0, 9]])
    x1 = np.random.multivariate_normal(mean1, cov, n1).transpose()
    x2 = np.random.multivariate_normal(mean2, cov, n2).transpose()
    return x1, x2

x1,x2 = generate_sample(600,0.1)

fig = plt.figure()

plt.scatter(x1[0,:], x1[1,:], s=100, c="red")
plt.scatter(x2[0,:], x2[1,:], s=100, marker="*", c="blue")
# plt.scatter(x1_x, x1_y, s=100, c="red")

bd_x = np.ones(100)*np.log(9)/4.0
bd_y = np.linspace(-10, 10, 100)
plt.plot(bd_x, bd_y,color='green')
fig.savefig("20230419_task2.png")
