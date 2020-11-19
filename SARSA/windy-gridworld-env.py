from collections import defaultdict
import numpy as np

ROWS = 5
COLS = 5

def policy(env, state, ep, Q):
    if np.random.rand() > ep:
        get_action_with_max_val([Q[state, a] for a in env.act_space])
    else:
        return np.random.choice(env.act_space)

def get_action_with_max_val(action_vals):
    return np.random.choice(np.flatnonzero(action_vals == np.max(action_vals)))

class NormalGridWorld:
    def __init__(self):
        self.act_space = [0,1,2,3]
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        return (0,0)

    def step(self, action):
        self.x, self.y = self.transition(action)
        if self.x == 4 and self.y == 4:
            return (self.x, self.y), 0, True
        return (self.x, self.y), -1, False

    def transition(self, action):
        x = self.x
        y = self.y
        if action == 0:
            x-=1
        elif action == 1:
            x+=1
        elif action == 2:
            y-=1
        elif action == 3:
            y+=1
        if x >= 0 and x <5:
            self.x = x
        if y >=0 and y < 5:
            self.y = y
        return self.x, self.y

def evaluate_policy(env, Q):
    S = env.reset()
    path = [S]
    done = False
    while True:
        action = policy(env,S,0,Q)
        S, reward, done = env.step(action)
        path.append(S)
        print(path)
        if done:
            break
    return path

def sarsa(env, ep, alpha, n_episodes, gamma=1.0):
    Q = defaultdict(float)
    for _ in range(n_episodes):
        if(_%500 == 0):
            print(_)
        S = env.reset()
        A = policy(env, S, ep, Q)
        while True:
            S_, reward, done = env.step(A)
            A_ = policy(env, S_, ep, Q)
            Q[S,A] = Q[S,A] + alpha * (reward + gamma*Q[S_,A_] - Q[S,A])
            S = S_
            A = A_
            if done:
                break
    return Q

env = NormalGridWorld()
Q = sarsa(env, 0.1, 0.5, 10000)
print(Q)
# optimal_path = evaluate_policy(env, Q)
# print (optimal_path)
