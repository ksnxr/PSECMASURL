from env import Env
from agent import Agent
import numpy as np
from settings import number_agents, number_iterations, log_interval
import matplotlib.pyplot as plt
import pickle
import os
import shutil


def get_action_state(number):
    """
    Get the encoding of an action.

    :param number: the number representing action
    :return: the encoding of the action char
    """
    if number == 0:
        return np.array([1, 0])
    elif number == 1:
        return np.array([0, 1])
    else:
        raise Exception


def get_partner_states(actions, index):
    """
    Get the partner states, i.e. the most recent actions of all other agents.

    :param actions: the most recent actions of all the agents
    :param index: the index of the selected agent
    :return: the encoding of all but the selected agent's most recent actions
    """
    l = [get_action_state(actions[ind]) for ind in range(number_agents) if ind != index]
    return np.concatenate(l)

def run_agent(i, j, k, partner_states, env):
    return reward1, reward2


if __name__ == '__main__':
    # init
    env = Env(number_agents)
    agents = []
    for i in range(number_agents):
        agents.append(Agent(number_agents, i))
    AC = []
    AD = []
    TFT = []
    RevTFT = []
    ac = 0
    ad = 0
    tft = 0
    rev_tft = 0
    partner_states = [get_partner_states(env.last_actions, i) for i in range(number_agents)]

    # Run the process for number_iters times.
    for i in range(number_iterations):
        if i % log_interval == 0:
            print(f'Iteration {i} starts.')
        for j in range(10):
            partner_states = [get_partner_states(env.last_actions, k) for k in range(number_agents)]
            for k in range(number_agents):
                # partner selection
                selected_partner = agents[k].select_partner(partner_states[k], i, j)
                # game play
                selected_action1 = agents[k].select_action(get_action_state(env.last_actions[selected_partner]))
                selected_action2 = agents[selected_partner].select_action(get_action_state(env.last_actions[k]))
                reward1, reward2 = env.compute_reward(k, selected_partner, selected_action1, selected_action2)
                agents[k].last_reward = reward1
                agents[selected_partner].last_reward = reward2

        env.step(i)

        if i % log_interval == 0:
            for j in range(number_agents):
                kind = agents[j].check_kind()
                if kind == 'AC':
                    ac += 1
                elif kind == 'AD':
                    ad += 1
                elif kind == 'TFT':
                    tft += 1
                else:
                    rev_tft += 1
            AC.append(ac)
            AD.append(ad)
            TFT.append(tft)
            RevTFT.append(rev_tft)
            print(f'All C: {ac}, All D: {ad}, TFT: {tft}, RevTFT: {rev_tft}')
            ac = 0
            ad = 0
            tft = 0
            rev_tft = 0

    plt.figure(figsize=(20, 8))
    plt.title('Agents')
    x = [i for i in range(number_iterations) if i % log_interval == 0]
    plt.plot(x, AC, 'g', label="All C")
    plt.plot(x, AD, 'c', label="All D")
    plt.plot(x, TFT, 'b', label="TFT")
    plt.plot(x, RevTFT, 'm', label="Rev TFT")
    plt.legend(loc='upper right')
    plt.plot()
    if os.path.exists('cache'):
        shutil.rmtree('cache')
    os.mkdir('cache')
    with open('cache/ac.txt', 'wb') as f:
        pickle.dump(AC, f)
    with open('cache/ad.txt', 'wb') as f:
        pickle.dump(AD, f)
    with open('cache/tft.txt', 'wb') as f:
        pickle.dump(TFT, f)
    with open('cache/rev_tft.txt', 'wb') as f:
        pickle.dump(RevTFT, f)

    env.end()
