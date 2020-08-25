import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import dqn_learning_rate as LR


class PartnerMemory:
    def __init__(self):
        """
        Initialize the partner memory.
        """
        self.states = []
        self.actions = []
        self.next_states = []

    def store(self, state, action, next_state):
        """
        Store a partner transition.

        :param state: the state to be stored
        :param action: the action to be stored
        :param next_state: the next_state to be stored
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)

    def clear(self):
        """
        Clear the memory.
        """
        assert len(self.states) == 10
        self.states = []
        self.actions = []
        self.next_states = []


class ActionMemory:
    def __init__(self):
        """
        Initialize the action memory.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.length = 0

    def store(self, state, action, reward, next_state):
        """
        Store a action transition.

        :param state: the state to be stored
        :param action: the action to be stored
        :param reward: the reward to be stored
        :param next_state: the next_state to be stored
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.length += 1

    def clear(self):
        """
        Clear the memory.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.length = 0


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        """
        Initialize the neural network.

        :param n_states: the number of input states
        :param n_actions: the number of output states
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 256)
        self.out = nn.Linear(256, n_actions)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        """
        Feed forward the neural network.

        :param x: the input
        :return: the result
        """
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, partner_n_states, partner_n_actions, partner_epsilon, action_n_states, action_n_actions,
                 action_epsilon):
        """
        Initialize the DQN module.

        :param partner_n_states: number of states of the partner network
        :param partner_n_actions: number of actions of the partner network
        :param partner_epsilon: epsilon of the partner network
        :param action_n_states: number of states of the action network
        :param action_n_actions: number of actions of the action network
        :param action_epsilon: epsilon of the action network
        """
        self.partner_net = Net(partner_n_states, partner_n_actions)
        self.action_net = Net(action_n_states, action_n_actions)
        self.partner_memory = PartnerMemory()
        self.action_memory = ActionMemory()
        self.partner_optimizer = torch.optim.Adam(self.partner_net.parameters(), lr=LR)
        self.action_optimizer = torch.optim.Adam(self.action_net.parameters(), lr=LR)
        self.partner_loss_func = nn.MSELoss()
        self.action_loss_func = nn.MSELoss()
        self.partner_n_states = partner_n_states
        self.action_n_states = action_n_states
        self.partner_n_actions = partner_n_actions
        self.action_n_actions = action_n_actions
        self.partner_epsilon = partner_epsilon
        self.action_epsilon = action_epsilon
        self.GAMMA = 0.99

    def partner_choose(self, x):
        """
        Choose the agent's partner.

        :param x: the partner states
        :return: the selected partner
        """
        x = torch.unsqueeze(torch.as_tensor(x, dtype=torch.float), 0)
        if np.random.uniform() < 1 - self.partner_epsilon:  # greedy
            actions_value = self.partner_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:  # random
            action = np.random.randint(0, self.partner_n_actions)
        return action

    def action_choose(self, x, greedy=False):
        """
        Choose the agent's action.

        :param x: the partner states
        :param greedy: whether to act greedily
        :return: the selected action
        """
        x = torch.unsqueeze(torch.as_tensor(x, dtype=torch.float), 0)
        if np.random.uniform() < 1 - self.action_epsilon or greedy:  # greedy
            actions_value = self.action_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:  # random
            action = np.random.randint(0, self.action_n_actions)
        return action

    def learn(self):
        """
        Train the neural networks using transitions stored in action memory and partner memory.
        """
        b_s = torch.as_tensor(np.vstack(self.partner_memory.states), dtype=torch.float)
        b_a = torch.as_tensor(np.vstack(self.partner_memory.actions), dtype=torch.long)
        b_s_ = torch.as_tensor(np.vstack(self.partner_memory.next_states), dtype=torch.float)
        partner_q_eval = self.partner_net(b_s).gather(1, b_a)
        partner_q_next = self.action_net(b_s_).detach()
        partner_q_target = self.GAMMA * partner_q_next.max(1)[0].view(10, 1)

        b_s = torch.as_tensor(np.vstack(self.action_memory.states), dtype=torch.float)
        b_a = torch.as_tensor(np.vstack(self.action_memory.actions), dtype=torch.long)
        b_r = torch.as_tensor(np.vstack(self.action_memory.rewards), dtype=torch.float)
        action_q_eval = self.action_net(b_s).gather(1, b_a)
        temp = []
        for next_state in self.action_memory.next_states:
            s_ = torch.as_tensor(next_state, dtype=torch.float)
            if len(s_) == 2:
                value = self.action_net(s_).detach().max().view(1)
            else:
                value = self.partner_net(s_).detach().max().view(1)
            temp.append(value)
        temp = torch.cat(temp, 0).view(self.action_memory.length, 1)
        action_q_target = b_r + self.GAMMA * temp

        partner_loss = self.partner_loss_func(partner_q_eval, partner_q_target)
        self.partner_optimizer.zero_grad()
        partner_loss.backward()
        self.partner_optimizer.step()
        self.partner_memory.clear()

        action_loss = self.action_loss_func(action_q_eval, action_q_target)
        self.action_optimizer.zero_grad()
        action_loss.backward()
        self.action_optimizer.step()
        self.action_memory.clear()
