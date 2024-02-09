import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import pandas as pd

data = loadmat('digit.mat')
train = data['X']
test = data['T']

x=train[:, 15, 9]
x = x.reshape([16, 16])*256
cv2.imwrite('gray.jpg',x)

mu_list = []
S = 0
for i in range(10):
    mu_list.append(np.mean(train[:, :, i], axis=1))
    S += np.cov(train[:, :, i])

t = test[:, :, 1]
invS = np.linalg.inv(S + 0.000001 * np.identity(256))

max_list = []
for i in range(10):
    t = test[:, :, i]
    p_array = np.empty((0,200))
    for j in range(10):
        p = mu_list[j][None, :].dot(invS).dot(t) - mu_list[j][None, :].dot(invS).dot(mu_list[j][:, None]) / 2
        p_array = np.append(p_array, p, axis=0)

    max_list.append(np.argmax(p_array,axis=0))

acc_array = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        acc_array[i,j] = np.sum(max_list[i] == j)

acc_num = 0
for i in range(10):
    acc_num += acc_array[i,i]

accuracy = acc_num/2000

df = pd.DataFrame(acc_array)
df.index = ['1','2','3','4','5','6','7','8','9','0']
df.columns = ['1','2','3','4','5','6','7','8','9','0']

print('線形判別分析による実験結果')
print('\n')
print('正答率：',accuracy)
print('\n')
print('縦が正解カテゴリ')
print('横が予測カテゴリ')
print('\n')
print(df)



       
