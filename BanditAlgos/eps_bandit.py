import numpy as np
from modules.plot_utility import *

k = 10
ITER = 1000
EPISODES = 1000

class eps_bandit:
    def __init__(self, k, eps, iter, mu='random'):
        self.k = k #no. of arms
        self.eps = eps #search probability
        self.iter = iter #no. of iterations
        self.n = 0 #step count
        self.k_n = np.zeros(k) #step count for each arm
        self.mean_reward = 0 #total mean reward
        self.reward = np.zeros(iter)
        self.k_reward = np.zeros(k) #mean reward for each arm

        if type(mu) == list or type(mu).__module__ == np.__name__:
            self.mu = np.array(mu)
        elif mu == 'random':
            #draw mean from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            #increase the mean for each arm by 1
            self.mu = np.linspace(0, k-1, k)

    def pull(self):
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.k)
        elif p < self.eps:
            #randomly select an action
            a = np.random.choice(self.k)
        else:
            #take greedy action
            a = np.argmax(self.k_reward)

        reward = np.random.normal(self.mu[a], 1)

        #update counts
        self.n += 1
        self.k_n[a] += 1

        #update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward)/self.n

        #update results for choosen bandit
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a])/self.k_n[a]

    def run(self):
        for i in range(self.iter):
            self.pull()
            self.reward[i] = self.mean_reward


    def reset(self):
        #resets results while keeping setting
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(iter)
        self.k_reward = np.zeros(k)

def mu_random(eps_val, mu=None):
    eps_rewards = np.zeros(ITER)
    #Run Experiments
    for i in range(EPISODES):
        #Initialize bandits
        if(mu is None):
            eps = eps_bandit(k, eps_val, ITER)
        else:
            eps = eps_bandit(k, eps_val, ITER, mu)

        #Run Experiments
        eps.run()

        #update the long term averages
        eps_rewards = eps_rewards + (eps.reward - eps_rewards)/ (i+1)

    return (eps,eps_rewards)

def mu_sequence(eps_val, mu=None):
    eps_rewards = np.zeros(ITER)
    eps_selection = np.zeros(k)

    #Run Experiments
    for i in range(EPISODES):
        #Initialize bandits
        if(mu is None):
            eps = eps_bandit(k, eps_val, ITER, mu='sequence')
        else:
            eps = eps_bandit(k, eps_val, ITER, mu)

        #Run Experiments
        eps.run()

        #Update long term average
        eps_rewards = eps_rewards + (eps.reward - eps_rewards)/ (i+1)

        #Average actions per episode
        eps_selection = eps_selection + (eps.k_n - eps_selection)/ (i+1)

    return [eps, eps_rewards, eps_selection]
