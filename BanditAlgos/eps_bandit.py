import numpy as np

k = 10
ITER = 1000
EPISODES = 1000

class eps_bandit:
    def __init__(self, eps):
        self.k = k #no. of arms
        self.eps = eps #search probability
        self.iter = ITER #no. of iterations
        self.n = 0 #step count
        self.k_n = np.zeros(k) #step count for each arm
        self.mean_reward = 0 #total mean reward
        self.reward = np.zeros(ITER)
        self.k_reward = np.zeros(k) #mean reward for each arm
        self.mu = np.random.normal(0, 1, k)


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

def start(eps_val):
    eps_rewards = np.zeros(ITER)
    #Run Experiments
    for i in range(EPISODES):
        #Initialize bandits
        eps = eps_bandit(eps_val)
        #Run Experiments
        eps.run()

        #update the long term averages
        eps_rewards = eps_rewards + (eps.reward - eps_rewards)/ (i+1)

    return (eps,eps_rewards)
