import gym
import numpy as np
import time


def init_values(s,a,type="random"):
    if type == "ones":
        return np.ones((s,a))
    elif type == "random":
        return np.random.random((s,a))
    elif type == "zeros":
        return np.zeros((s,a))

def epsilon_action(Q, epsilon, n_actions, s, train=False):
    if train or np.random.random() < epsilon:
        return np.argmax(Q[s,:])
    else:
        return np.random.randint(0, n_actions)

def expected_sarsa(alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
    env = gym.make('Taxi-v3')
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_values(n_states, n_actions, type="ones")
    timestep_reward = []

    for episode in range(episodes):
        print(f"Episode: {episode}")
        total_reward = 0
        s = env.reset()
        t = 0
        while t < max_steps:
            if render:
                env.render()
            t+=1
            a = epsilon_action(Q, epsilon, n_actions, s)
            s_, reward, done, info = env.step(a)
            total_reward += reward
            if done:
                Q[s,a] += alpha * (reward - Q[s,a])
            else:
                expected_value = np.mean(Q[s_,:])
                Q[s,a] += alpha * (reward + (gamma * expected_value) - Q[s,a])
            s = s_
            if done:
                if True:
                    print(f"This episode took {t} timesteps and reward {total_reward}")
                timestep_reward.append(total_reward)
                break
    if render:
        print(f"Here are the Q values:\n{Q}\nTesting now:")
    if test:
        test_agent(Q, env, n_tests, n_actions)
    return timestep_reward

def test_agent(Q, env, n_tests, n_actions, delay=0.1):
    for test in range(n_tests):
        print(f"Test #{test}")
        s = env.reset()
        done = False
        epsilon = 0
        total_reward = 0
        while True:
            time.sleep(delay)
            env.render()
            a = epsilon_action(Q, epsilon, n_actions, s, train=True)
            print(f"Chose action {a} for state {s}")
            s, reward, done, info = env.step(a)
            total_reward += reward
            if done:
                print(f"Episode reward: {total_reward}")
                time.sleep(1)
                break

if __name__ == '__main__':
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9
    epsiodes =  1000
    max_steps = 2500
    n_tests = 20
    timestep_reward = expected_sarsa(alpha, gamma, epsilon, epsiodes, max_steps, n_tests, render=False, test=True)
    print(timestep_reward)
