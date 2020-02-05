import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import time
import math
import copy

from gym.wrappers import Monitor

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CartPoleAI(nn.Module):
    def __init__(self):
        super.__init()
        self.fc = nn.Sequential(
                nn.Linear(4,128,bias = True),
                nn.ReLU(),
                nn.Linear(128,2,bias=True),
                nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x

def init_weights(m):
    if(type(m)==nn.Linear) or (type(m)==nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = CartPoleAI()

        for param in agent.parameters():
            param.requires_grad = False

        init_weights(agent)
        agents.append(agent)

    return agents
    
