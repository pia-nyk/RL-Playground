import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def line_plot(eps_0_rewards, eps_01_rewards, eps_1_rewards, iter, episodes, k=0, eps_0=None):
    plt.figure(figsize=(12,8))
    plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
    plt.plot(eps_01_rewards, label="$\epsilon=0.01$")
    plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
    for i in range(k):
        plt.hlines(eps_0.mu[i], xmin=0, xmax=iter, alpha=0.5, linestyle="--")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon-greedy$ Rewards after " + str(episodes) + " Episodes")
    plt.show()

def bar_plot(eps_0_selection, eps_01_selection, eps_1_selection, k):
    bins = np.linspace(0, k-1, k)
    plt.figure(figsize=(12, 8))
    plt.bar(bins, eps_0_selection, width=0.33, color='b', label="$\epsilon=0$")
    plt.bar(bins+0.33, eps_01_selection, width=0.33, color='g', label="$\epsilon=0.01$")
    plt.bar(bins+0.66, eps_1_selection, width=0.33, color='r', label="$\epsilon=0.1$")
    plt.legend()
    plt.xlim([0,k])
    plt.title("Action Selected by each algorithm")
    plt.xlabel("Action")
    plt.ylabel("Number of Actions Taken")
    plt.show()

def percent_actions(eps_0_selection, eps_01_selection, eps_1_selection, iter, k):
    opt_per = np.array([eps_0_selection, eps_01_selection, eps_1_selection])/ iter * 100
    df = pd.DataFrame(opt_per, index=['$\epsilon=0$', '$\epsilon=0.01$', '$\epsilon=0.1$'], columns=["a = " + str(x) for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df.to_string())
