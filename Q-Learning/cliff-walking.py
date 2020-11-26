from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

'''
    epsilon greedy policy, chooses randomly with probability eps,
    chooses greedily otherwise
'''
def policy(env, st, eps, Q):
    if np.random.rand() > eps:
        return get_action_with_max_val([Q[st,a] for a in env.act_space])
    else:
        return np.random.choice(env.act_space)

'''
    provided a list of q-values of actions possible from a state, this function will return the
    action with the max q-value
'''
def get_action_with_max_val(action_vals):
    return np.random.choice(np.flatnonzero(action_vals == np.max(action_vals)))

'''
    provided a list of q-values of actions possible from a state, this function will return the
    max q-value
'''
def get_max_action_val(action_vals):
    return np.max(action_vals)

'''
    grid world with a start and end state, few states in between which need to be
    avoided. Actions possible - up, down, right, left (Refer Sutton-Barto)
'''
class CliffWalk:
    def __init__(self):
        self.act_space = [0,1,2,3]
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        return (0,0) #start state

    def step(self, action):
        self.x, self.y = self.transition(action)
        if self.x == 11 and self.y == 0: #terminal state
            return (self.x, self.y), 0, True
        elif self.x >= 1 and self.x <= 10 and self.y == 0: #states to avoid
            return (0,0), -100, False
        else:
            return (self.x, self.y), 0, False

    def transition(self, action):
        x = self.x
        y = self.y
        if action == 0:
            x-=1 #left
        elif action == 1:
            x+=1 #right
        elif action == 2:
            y-=1 #up
        elif action == 3:
            y+=1 #down
        if x >= 0 and x <= 11: #horizontal bounds
            self.x = x
        if y >= 0 and y <= 3: #vertical bounds
            self.y = y
        return self.x, self.y


def Q_learning(env, n_episodes, gamma, alpha, eps):
    Q = defaultdict(float)
    rewards = []
    for _ in range(n_episodes):
        if(_%500 == 0):
            print(_)
        rewards.append(0)
        S = env.reset()
        while True:
            A = policy(env, S, eps, Q)
            S_, R, done = env.step(A)
            Q[S,A] = Q[S,A] + alpha * (R + gamma * get_max_action_val([Q[S,a] for a in env.act_space]) - Q[S,A]) #update step
            S = S_
            rewards[-1]+=R
            if done:
                break
    return Q, rewards

env = CliffWalk()
Q, rewards = Q_learning(env, 10000, 1.0, 0.5, 0.1)
