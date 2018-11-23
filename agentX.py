#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:16:32 2018

@author: helgi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import torch
from torch.autograd import Variable
import Backgammon
import flipped_agent
device = torch.device('cpu')


# load to global memory
w1 = torch.load('X_w1_trained_99.pth')
w2 = torch.load('X_w2_trained_99.pth')
b1 = torch.load('X_b1_trained_99.pth')
b2 = torch.load('X_b2_trained_99.pth')
nx = 24 * 2 * 6 + 4

def one_hot_encoding(board, nSecondRoll):
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1)
    # where are the zero, single, double, ... discs
    for i in range(0,5):
        oneHot[i*24+np.where(board[1:25] == i)[0]-1] = 1
    # anything above 4 should be also labelled
    oneHot[5*24+np.where(board[1:25] >= 5)[0]-1] = 1
    # now repeat the process but for other player "-1"
    for i in range(0,5):
        oneHot[6*24+i*24+np.where(board[1:25] == -i)[0]-1] = 1
    # anything above 4 should be also labelled
    oneHot[6*24+5*24+np.where(board[1:25] <= -5)[0]-1] = 1
    # now add the jail and home bits
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = nSecondRoll
    return oneHot


# this epsilon greedy policy uses a feed-forward neural network to approximate the after-state value function
def action_test(board, dice, oplayer, i = 0):

    flippedplayer = -1
    if (flippedplayer == oplayer): # view it from player 1 perspective
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer # player now the other player +1
    else:
        player = oplayer
    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
    na = len(possible_boards)
    if (na == 0):
        return []
    xa = np.zeros((na,nx+1))
    va = np.zeros((na))
    for j in range(0, na):
        xa[j,:] = one_hot_encoding(possible_boards[j],i)
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu()
    action = possible_moves[np.argmax(va)]
    if (flippedplayer == oplayer): # map this move to right view
        action = flipped_agent.flip_move(action)
    return action
