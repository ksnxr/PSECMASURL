import matplotlib
import matplotlib.pyplot as plt
from settings import number_iterations, log_interval
import pickle
import random

matplotlib.use('TkAgg')


class Env:
    def __init__(self, total):
        """
        Initialize.

        :param total: the number of agents
        """
        self.last_actions = [random.choice([0, 1]) for _ in range(total)]
        self.mutual_cooperation = []
        self.mutual_defection = []
        self.exploitation = []
        self.deception = []
        self.societal_reward = []
        self.mc = 0
        self.md = 0
        self.ex = 0
        self.de = 0
        self.sr = 0
        self.total = 10 * total
        self.T = 4
        self.R = 3
        self.P = 1
        self.S = 0

    def step(self, i):
        """
        Go on to the next step.

        :return: the most recent action of the agents
        """
        if i % log_interval == 0:
            print(f'Mutual cooperation: {self.mc}, mutual defection: {self.md}, exploitation: {self.ex}, deception: {self.de}, societal reward: {self.sr}')
        self.mutual_cooperation.append(self.mc / self.total)
        self.mutual_defection.append(self.md / self.total)
        self.exploitation.append(self.ex / self.total)
        self.deception.append(self.de / self.total)
        self.societal_reward.append(self.sr)
        self.mc = 0
        self.md = 0
        self.ex = 0
        self.de = 0
        self.sr = 0

    def end(self):
        """
        End the process and plot the figure.
        """
        with open('cache/mc.txt', 'wb') as f:
            pickle.dump(self.mutual_cooperation, f)
        with open('cache/md.txt', 'wb') as f:
            pickle.dump(self.mutual_defection, f)
        with open('cache/ex.txt', 'wb') as f:
            pickle.dump(self.exploitation, f)
        with open('cache/de.txt', 'wb') as f:   
            pickle.dump(self.deception, f)
        with open('cache/sr.txt', 'wb') as f:
            pickle.dump(self.societal_reward, f)
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(111)
        x = [_ for _ in range(number_iterations)]
        ax1.plot(x, self.mutual_cooperation, 'g', label="Mutual Cooperation")
        ax1.plot(x, self.mutual_defection, 'c', label="Mutual Defection")
        ax1.plot(x, self.exploitation, 'b', label="Exploitation")
        ax1.plot(x, self.deception, 'm', label="Deception")
        plt.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.plot(x, self.societal_reward, 'r', label="Societal Reward")
        plt.title("Games")
        plt.legend(loc='lower right')
        plt.show()

    def compute_reward(self, number1, number2, action1, action2):
        """
        With agent number1 doing action1, agent number2 doing action2, compute their respective rewards.

        :param number1: the index of the first agent
        :param number2: the index of the second agent
        :param action1: the first agent's action
        :param action2: the second agent's action
        :return: the action rewards of agents
        """
        self.last_actions[number1] = action1
        if action1 == 0 and action2 == 0:
            self.mc += 1
            self.sr += (self.R + self.R)
            return self.R, self.R
        elif action1 == 0 and action2 == 1:
            self.de += 1
            self.sr += (self.S + self.T)
            return self.S, self.T
        elif action1 == 1 and action2 == 0:
            self.ex += 1
            self.sr += (self.T + self.S)
            return self.T, self.S
        elif action1 == 1 and action2 == 1:
            self.md += 1
            self.sr += (self.P + self.P)
            return self.P, self.P
        else:
            raise Exception
