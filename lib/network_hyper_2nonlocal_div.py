from torch import nn
import numpy as np
import torch
# from lib.non_local_concatenation import NONLocalBlock2D
# from lib.non_local_gaussian import NONLocalBlock2D
# from lib.non_local_embedded_gaussian_new4 import NONLocalBlock3D
from lib.non_local_embedded_gaussian_new4_1 import NONLocalBlock1D
# from lib.non_local_dot_product import NONLocalBlock2D
from lib.add_neighbor import neighbor_add
# #############################################################################
# this network is built for hyperspectral image classification
class Network_div(nn.Module):
    def __init__(self):
        super(Network_div, self).__init__()

        self.nl_4 = NONLocalBlock1D(in_channels=1)


    def forward(self, output, indices_o, indices, cube_size, gt):

        new_output = output[0:1, :]

        pos_o = indices_o.cpu().numpy()
        pos = indices.cpu().numpy()
        gt = gt.cpu().numpy()

        for i in range(indices_o.shape[0]):
            temp_output = output[0:1, :]
            row, col = pos_o[i]
            new_gt = neighbor_add(row, col,cube_size, gt)
            tp1, tp2 = np.nonzero(new_gt)
            mark = 0
            for k in range(tp1.shape[0]):
                x1 = tp1[k]
                y1 = tp2[k]
                for j in range(indices.shape[0]):
                    x2, y2 = pos[j]
                    if x1 == x2 and y1 == y2:
                        temp_output = torch.cat((temp_output, output[j:(j+1), :]), 0)
                        if x2 == row and y2 == col:
                            mark = temp_output.shape[0]-1

            bs = temp_output.shape[0]
            temp_output1 = temp_output[1: bs, :]
            bs = bs-1

            if bs < 2:
                lis = pos.tolist()
                ind = lis.index([row, col])
                new_output = torch.cat((new_output, output[ind, :].unsqueeze(0)), 0)
            else:
                temp_output1 = temp_output1.view(bs, 1, -1)
                temp_output1 = self.nl_4(temp_output1)
                temp_output1 = temp_output1.reshape(bs, 9)


                new_output = torch.cat((new_output, temp_output1[mark - 1, :].unsqueeze(0)), 0)

        ll = new_output.shape[0]
        new_output = new_output[1: ll, :]

        return new_output


    # def forward_with_nl_map(self, output, indices_o, indices, cube_size, gt):
    #
    #     new_output = output[0:1, :]
    #
    #     pos_o = indices_o.cpu().numpy()
    #     pos = indices.cpu().numpy()
    #     gt = gt.cpu().numpy()
    #
    #     for i in range(indices_o.shape[0]):
    #         temp_output = output[0:1, :]
    #         row, col = pos_o[i]
    #         new_gt = neighbor_add(row, col, cube_size, gt)
    #         tp1, tp2 = np.nonzero(new_gt)
    #         mark = 0
    #         for k in range(tp1.shape[0]):
    #             x1 = tp1[k]
    #             y1 = tp2[k]
    #             for j in range(indices.shape[0]):
    #                 x2, y2 = pos[j]
    #                 if x1 == x2 and y1 == y2:
    #                     temp_output = torch.cat((temp_output, output[j:(j + 1), :]), 0)
    #                     if x2 == row and y2 == col:
    #                         mark = temp_output.shape[0] - 1
    #
    #         bs = temp_output.shape[0]
    #         temp_output1 = temp_output[1: bs, :]
    #         bs = bs - 1
    #
    #         if bs < 2:
    #             lis = pos.tolist()
    #             ind = lis.index([row, col])
    #             new_output = torch.cat((new_output, output[ind, :].unsqueeze(0)), 0)
    #         else:
    #             temp_output1 = temp_output1.view(bs, 1, -1)
    #             temp_output1, nl_map = self.nl_4(temp_output1, return_nl_map=True)
    #
    #             temp_output1 = temp_output1.reshape(bs, 9)
    #
    #             new_output = torch.cat((new_output, temp_output1[mark - 1, :].unsqueeze(0)), 0)
    #
    #     ll = new_output.shape[0]
    #     new_output = new_output[1: ll, :]
    #
    #     return new_output, [nl_map]




