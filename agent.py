from dqn import DQN
import numpy as np
from settings import debug, debug_agent


class Agent:
    def __init__(self, total, identity):
        """
        Initialize the agent.

        :param total: total number of agents
        :param identity: the id of the current agent
        """
        self.identity = identity
        self.dqn = DQN(2 * total - 2, total - 1, 0.1, 2, 2, 0.05)
        self.others = [i for i in range(total) if i != self.identity]
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def select_partner(self, partner_states, i, j):
        """
        Let the agent select its partner from other agents based on selection_states.

        :param partner_states: the most recent actions of all other agents
        :param i: the iteration
        :param j: the sub-iteration
        :return: the id of the agent's partner
        """
        if self.last_reward is not None:
            self.dqn.action_memory.store(self.last_state, self.last_action, self.last_reward,
                                         partner_states)
            if debug and self.identity == debug_agent:
                print(self.last_state, self.last_action, self.last_reward,
                      partner_states)
        if i != 0 and j == 0:
            self.dqn.learn()
        self.last_reward = None
        p = self.dqn.partner_choose(partner_states)
        self.dqn.partner_memory.store(partner_states, p, partner_states[2 * p: 2 * p + 2])
        if debug and self.identity == debug_agent:
            print(partner_states, p, partner_states[2 * p: 2 * p + 2])
        return self.others[p]

    def select_action(self, action_state):
        """
        Let the agent select its action in the game based on action_state.

        :param action_state: the most recent action of the agent's current opponent
        :return: the agent's action in the game
        """
        if self.last_reward is not None:
            self.dqn.action_memory.store(self.last_state, self.last_action, self.last_reward, action_state)
            if debug and self.identity == debug_agent:
                print(self.last_state, self.last_action, self.last_reward,
                      action_state)
        a = self.dqn.action_choose(action_state)
        self.last_state = action_state
        self.last_action = a
        return a

    def check_kind(self):
        """
        Check the kind of the agent.

        :return: the kind of the agent
        """
        action1 = self.dqn.action_choose(np.array([1, 0]), greedy=True)  # 'C'
        action2 = self.dqn.action_choose(np.array([0, 1]), greedy=True)  # 'D'
        if action1 == 0 and action2 == 0:
            return 'AC'
        elif action1 == 1 and action2 == 1:
            return 'AD'
        elif action1 == 0 and action2 == 1:
            return 'TFT'
        else:
            return 'RevTFT'
