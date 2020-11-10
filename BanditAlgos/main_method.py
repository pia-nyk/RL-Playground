import eps_bandit as egb
import UCB_bandit as ucbandit
import matplotlib.pyplot as plt

EPISODES = 1000

def epsilon_greedy_cmp():
    eps1, eps1rewards = egb.start(0.1)
    eps2, eps2rewards = egb.start(0)
    eps3, eps3rewards = egb.start(0.01)

    plt.figure(figsize=(12,8))
    plt.plot(eps1rewards, label="$\epsilon=0.1$")
    plt.plot(eps2rewards, label="$\epsilon=0$")
    plt.plot(eps3rewards, label="$\epsilon=0.01$")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average greedy, " +
        "$\epsilon=0.1$," + "$\epsilon=0.01$" + " Rewards after "
        + str(EPISODES) + " Episodes")
    plt.show()

def ucb_epsilon_cmp():
    ucb, ucbrewards = ucbandit.start(2)
    eps1, eps1rewards = egb.start(0.1)

    plt.figure(figsize=(12,8))
    plt.plot(ucbrewards, label="$\c=2$")
    plt.plot(eps1rewards, label="$\epsilon=0.1$")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average UCB, " +
        "$\c=2$," + "$\epsilon=0.1$" + " Rewards after "
        + str(EPISODES) + " Episodes")
    plt.show()

if __name__ == '__main__':
    ucb_epsilon_cmp()
