import numpy as np
import matplotlib.pyplot as plt

def MLE(N):
    sigma_list = []
    for i in range(1000):
        #平均10，分散400の正規分布からデータを10個生成
        data_arr = np.random.normal(loc=10, scale=20, size=N)
        ave = np.sum(data_arr)/N

        #分散の最尤推定量
        sigma = (np.linalg.norm(data_arr - ave, ord=2))**2/N
        sigma_list.append(sigma)

    #分散の最尤推定量の平均値
    ave_sigma = sum(sigma_list)/len(sigma_list)
    #分散の最尤推定量の平均値と実際の分散との誤差
    error_sigma = np.abs(ave_sigma - 400)
    print(f'ave_sigma={ave_sigma}')
    print(f'error_sigma+{error_sigma}\n')
    
    
    return sigma_list

N_list = [10, 100, 1000, 10000, 100000]
hist_list = []
for j in N_list:
    plt.hist(MLE(j), label=f'N={j}')

plt.legend(loc='upper right')
plt.savefig('./sigma_histogram')