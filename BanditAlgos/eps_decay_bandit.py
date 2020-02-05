import numpy as np
from modules.plot_utility import *

k = 10
ITER = 1000
EPISODES = 1000

class eps_decay_bandit:
    def __init__(self, k, iter, mu='random'):
        #total no. of bandits
        self.k = k
        #total no. of iterations
        self.iter = iter
        #step count
        self.n = 0
        #step count for each bandit
        self.k_n = np.zeros(k)
        #total mean reward
        self.mean_reward = 0
        #reward in each iteration
        self.reward = np.zeros(iter)
        #mean reward for each bandit
        self.k_reward = np.zeros(k)

        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)

    def pull(self):
        p = np.random.rand()
        if p < 1/(1 + self.n / self.k):
            #Randomly select an action
            a = np.random.choice(self.k)
        else:
            #Choose the greedy action
            a = np.argmax(self.k_reward)

        reward = np.random.normal(self.mu[a], 1)

        #Update counts
        self.n += 1
        self.k_n[a] += 1

        #Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        #Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]

    def run(self):
        for i in range(self.iter):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self):
        #Reset results while keeping setting
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iter)
        self.k_reward = np.zeros(k)

def mu_random(mu=None):
    eps_rewards = np.zeros(ITER)
    #Run Experiments
    for i in range(EPISODES):
        #Initialize bandits
        if(mu is None):
            eps = eps_decay_bandit(k, ITER)
        else:
            eps = eps_decay_bandit(k, ITER, mu)

        #Run Experiments
        eps.run()

        #update the long term averages
        eps_rewards = eps_rewards + (eps.reward - eps_rewards)/ (i+1)

    return (eps,eps_rewards)
