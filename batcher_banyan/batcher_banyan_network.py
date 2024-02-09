# 1->down 0->up
updown = [[1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0],
          [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

#入ってくるportの番号
batcher_connect = [[0, 2, 1, 3, 4, 6, 5, 7], [0, 2, 1, 3, 4, 6, 5, 7],
                   [0, 4, 1, 5, 2, 6, 3, 7], [0, 4, 2, 6, 1, 5, 3, 7],
                   [0, 2, 1, 3, 4, 6, 5, 7], [0, 4, 1, 5, 2, 6, 3, 7]]

banyan_connect = [[0, 4, 2, 6, 1, 5, 3, 7], [0, 2, 1, 3, 4, 6, 5, 7],
                  [0, 1, 2, 3, 4, 5, 6, 7]]

def batcher_network(in_list, layer_num):
    new_list = [i if i is not None else 100 for i in in_list]
    out_list = [0]*8
    next_in_list = [0]*8

    for j in range(4):
        if updown[layer_num][j] == 1:
            if new_list[j*2] > new_list[j*2+1]:
                out_list[j*2] = new_list[j*2+1]
                out_list[j*2+1] = new_list[j*2]
            else :
                out_list[j*2] = new_list[j*2]
                out_list[j*2+1] = new_list[j*2+1]

        elif updown[layer_num][j] == 0:
            if new_list[j*2] < new_list[j*2+1]:
                out_list[j*2] = new_list[j*2+1]
                out_list[j*2+1] = new_list[j*2]
            else:
                out_list[j*2] = new_list[j*2]
                out_list[j*2+1] = new_list[j*2+1]

    for i in range(8):
        next_in_list[i] = out_list[batcher_connect[layer_num][i]]
    
    return next_in_list

def banyan_network(in_list, bit_num):
    out_list = [0]*8
    next_in_list = [0]*8
    
    for i in range(4):
        if in_list[i*2] != None:
            if in_list[i*2][bit_num] == '0':
                out_list[i*2] = in_list[i*2]
            elif in_list[i*2][bit_num] == '1':
                out_list[i*2+1] = in_list[i*2]

        if in_list[i*2+1] != None:
            if in_list[i*2+1][bit_num] == '0':
                out_list[i*2] = in_list[i*2+1]
            elif in_list[i*2+1][bit_num] == '1':
                out_list[i*2+1] = in_list[i*2+1]
    
    out_list = [j if j != 0 else None for j in out_list]

    for k in range(8):
        next_in_list[k] = out_list[banyan_connect[bit_num][k]]
    
    return next_in_list

def batcher_banyan(input):
    for i in range(6):
        input = batcher_network(input, i)

    banyan_in = [0]*8
    for i in range(8):
            if input[i] > 7:
                banyan_in[i] = None
            else:
                banyan_in[i] = format(input[i], f'0{3}b')

    for j in range(3):
        banyan_in = banyan_network(banyan_in, j)

    output = []
    for i in range(8):
        if banyan_in[i] == None:
            output.append(None)
        else:
            output.append(int(banyan_in[i], 2))

    return output

input1 = [None, 5, None, None, None, None, 7, 3]
input2 = [6, None, None, None, 7, None, None, 2]
input3 = [7, 2, 4, 6, 5, 1, 3, 0]

output1 = batcher_banyan(input1)
output2 = batcher_banyan(input2)
output3 = batcher_banyan(input3)







            

