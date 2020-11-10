import numpy as np
import math
import sys

k = 10
ITER = 1000
EPISODES = 1000

class UCB_bandit:
    def __init__(self, c):
        self.k = k
        self.iter = ITER
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(ITER)
        self.k_reward = np.zeros(k)
        self.mu = np.random.normal(0, 1, k)
        self.ucb = np.zeros(k)
        self.c = c

    def pull(self):
        for i in range(k):
            if self.k_n[i] == 0:
                self.ucb[i] = sys.maxsize
            else:
                self.ucb[i] = self.mu[i] + (self.c * math.sqrt(math.log(self.n)/self.k_n[i]))
        a = np.argmax(self.ucb)
        reward = np.random.normal(self.mu[a], 1)
        self.n += 1
        self.k_n[a] += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward)/self.n
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a])/self.k_n[a]

    def run(self):
        for i in range(self.iter):
            self.pull()
            self.reward[i] = self.mean_reward

def start(c):
    ucb_rewards = np.zeros(ITER)
    for i in range(EPISODES):
        ucb = UCB_bandit(c)
        ucb.run()
        ucb_rewards = ucb_rewards + (ucb.reward - ucb_rewards)/ (i+1)
    return (ucb,ucb_rewards)
