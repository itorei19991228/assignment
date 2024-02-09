import numpy as np
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt

data = loadmat('./digit.mat')
train = data['X']
test = data['T']

train = train.reshape([256, 5000]).T
train_label = np.tile(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), 500)

test = test.reshape([256, 2000]).T
test_label = np.tile(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), 200)
    
def distance(X, Y):
    Z = torch.zeros((X.shape[0], Y.shape[0]))
    for n in range(X.shape[0]):
        for m in range(Y.shape[0]):
            Z[n][m] = torch.linalg.norm(X[n] - Y[m])

    return Z

def prediction(train, test, k, train_label):
    distance_matrix = distance(test, train)
    sort_index = torch.argsort(distance_matrix, axis=1) 
    nearest_k = sort_index[:,:k] 
    labels = train_label[nearest_k] 
    label_num = torch.sum(torch.eye(10)[labels], axis=1) 
    Y = torch.argmax(label_num, axis=1) 

    return Y

def get_accuracy(pred, real):
    accuracy = torch.sum(torch.ones(pred.shape)[pred == real])/pred.shape[0]
    return accuracy

n = train.shape[0]
t = 10

k_list = range(1,21)
all = np.arange(0, n)
loss_list = []
for k in k_list:
    loss = 0
    for batch in range(t):
        test_part = np.arange(batch * (n//t), batch * (n//t) + n//t)
        train_part = np.delete(all, test_part)

        test_x = train[test_part, :]
        test_label_x = train_label[test_part]
        train_x = train[train_part, :]
        train_label_x = train_label[train_part]

        test_x = torch.tensor(test_x)
        test_label_x = torch.tensor(test_label_x)
        train_x = torch.tensor(train_x)
        train_label_x = torch.tensor(train_label_x)

        test_x = test_x.to('cuda:0')
        test_label_x = test_label_x.to('cuda:0')
        train_x = train_x.to('cuda:0')
        train_label_x = train_label_x.to('cuda:0')

        y = prediction(train_x, test_x, k, train_label_x)
        y = y.to('cuda:0')
        accuracy = get_accuracy(y, test_label_x)
        loss += 1-accuracy
        print(f'k={k}, batch={batch}, error={loss}')
    
    loss_list.append(loss/t)

max_k_index = loss_list.index(min(loss_list))

max_k = k_list[max_k_index]

fig = plt.figure()
plt.plot(k_list, loss_list)
fig.savefig("./20230524_task2.png")

train = torch.tensor(train)
train_label = torch.tensor(train_label)
test = torch.tensor(test)
test_label = torch.tensor(test_label)

train = train.to('cuda:0')
train_label = train_label.to('cuda:0')
test = test.to('cuda:0')
test_label = test_label.to('cuda:0')

pred = prediction(train, test, max_k, train_label)
pred = pred.to('cuda:0')
accuracy = get_accuracy(pred, test_label)
print(max_k)
print(accuracy)
