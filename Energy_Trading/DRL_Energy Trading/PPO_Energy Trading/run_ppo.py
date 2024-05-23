from PPO import PPO
from Env import MyEnv
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from painter import painter
import pandas as pd
import time
import csv

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置宋体字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号 有中文出现的情况，需要u'内容'
EPSILON_DECAY = 40000
EPSILON_START = 0.8
EPSILON_END = 0.02


len_episode = 24
LR_A = 3e-4
LR_C = 5e-4
GAMMA = 0.99
EPSILON = 0.2
A_update_steps = 10
C_update_steps = 10
batch = 32

env = MyEnv()
n_states = 32
n_actions = env.action_space.n
num_ev = 10
count = int(num_ev / 10)
count_write = 0
write = np.zeros(count)
n_episode = 50000 * count



agent = PPO(n_states=n_states, n_actions=n_actions, lr_a=LR_A, lr_c=LR_C, gamma=GAMMA, epsilon=EPSILON,
            a_update_steps=A_update_steps,
            c_update_steps=C_update_steps)


all_ep_r = []
states, actions, rewards = [], [], []
m_count = 0
rewards1 = []

REWARD_BUFFER = np.empty(shape=n_episode)
write_r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s = env.reset()
states_PPO, actions_PPO, rewards_PPO = [], [], []


for episode_i in range(n_episode):
    epsilon = np.interp(episode_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    random_sample = random.random()
    if random_sample <= epsilon:
        a = env.action_space.sample()
    else:
        a = agent.choose_action(s)

    s_, r = env.step(a)
    states_PPO.append(s)
    actions_PPO.append(a)
    rewards_PPO.append(r)
    REWARD_BUFFER[episode_i] = r
    write[episode_i % count] = r
    if (episode_i + 1) % count == 0:
        write_r[count_write] = np.sum(write)
        count_write = (count_write + 1) % 30

    if (episode_i + 1) % batch == 0 or episode_i == n_episode - 1:
        states = np.array(states_PPO)
        actions = np.array(actions_PPO)
        rewards = np.array(rewards_PPO)

        targets = agent.discount_reward(rewards, s_)  # 奖励回溯
        agent.update(states, actions, targets)  # 进行actor和critic网络的更新
        states_PPO, actions_PPO, rewards_PPO = [], [], []

    s = s_
    if (episode_i + 1) % (count * 30) == 0:
        rewards1.append(np.mean(write_r))

    if episode_i % 100 == 0:
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))
        print(a, r)



