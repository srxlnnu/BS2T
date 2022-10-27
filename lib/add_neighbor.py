
import numpy as np



# test_pos = sio.loadmat('./pos/test_pos_indian.mat')
# test_pos = test_pos['test_pos']
# test_pos_1 = test_pos[0,0]
# for i in range (1, np.size(test_pos,1)):
#     test_pos_1 = np.vstack((test_pos_1,test_pos[0,i]))
# test_pos = test_pos_1

# new_train_gt = np.zeros_like(gt)
def neighbor_add(row, col,cube_size, gt):

    new_train_gt = np.zeros_like(gt)
    da = gt.shape[0]
    w_size = cube_size # 7
    t = w_size//2
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            if i + row < 0 or i + row >= da or j + col < 0 or j + col >= da:
                continue
            else:
                new_train_gt[i + row, j + col] = gt[i + row, j + col]
    return new_train_gt