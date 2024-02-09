import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 3000
#標本データ生成
def data_generate(n):
    x = np.zeros(n)
    u = np.random.rand(n)
    index1 = np.where((0 <= u) & (u < 1/8))
    x[index1] = np.sqrt(8 * u[index1])
    index2 = np.where((1/8 <= u) & (u < 1/4))
    x[index2] = 2 - np.sqrt(2 - 8 * u[index2])
    index3 = np.where((1/4 <= u) & (u < 1/2))
    x[index3] = 1 + 4 * u[index3]
    index4 = np.where((1/2 <= u) & (u < 3/4))
    x[index4] = 3 + np.sqrt(4 * u[index4] - 2)
    index5 = np.where((3/4 <= u) & (u <= 1))
    x[index5] = 5 - np.sqrt(4 - 4 * u[index5])

    return x

def kernel_func(x):
    return 1/np.sqrt(2 * np.pi) * np.exp(- (x ** 2)/2)

def p_x(x, x_i, h):
    n = x_i.shape[0]
    sum = 0
    for i in range(n):
        sum += kernel_func((x - x_i[i])/h)

    return sum/n/h

x = np.arange(0, 5.0, 0.05)
x_i = data_generate(n)

h_arr = np.arange(0.01, 0.5, 0.01)

t = 10
LCV_list = []
for i in range(h_arr.shape[0]):
    LCV = 0
    for batch in range(t):
        test_x = x_i[batch * (n//t) : batch * (n//t) + n//t]
        train_x = np.delete(x_i, np.arange(batch * (n//t), batch * (n//t) + n//t))
        y = p_x(test_x, train_x, h_arr[i])
        LCV += np.sum(np.log(y))/(n//t)
    
    LCV_list.append(LCV/t)

max_h_index = LCV_list.index(max(LCV_list))
max_h = h_arr[max_h_index]
fig1 = plt.figure()
plt.plot(h_arr, LCV_list)
plt.xlabel('h')
plt.ylabel('LCV')
fig1.savefig(f'./LCV.png')


h_list = [0.01, 0.1, max_h, 0.5]
for h in h_list:
    
    fig2 = plt.figure()
    plt.plot(x, p_x(x, x_i, h), label=f'h={h}')
    plt.legend(loc='upper right')
    fig2.savefig(f'./histgram_h={h}.png')
