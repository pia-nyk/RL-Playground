import eps_decay_bandit as edb
import eps_bandit as egb
import matplotlib.pyplot as plt

EPISODES = 1000

if __name__ == '__main__':
    eps_decay, eps_decay_rewards = edb.mu_random()
    eps_1_greedy, eps_1_rewards = egb.mu_random(0.1, eps_decay.mu.copy())

    plt.figure(figsize=(12,8))
    plt.plot(eps_decay_rewards, label="$\epsilon-decay$")
    plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon-decay$ and" +
        "$\epsilon-greedy$ Rewards after "
        + str(EPISODES) + " Episodes")
    plt.show()
