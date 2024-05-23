from agent import Agent
from Env import MyEnv

import numpy as np
import random
import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from painter import painter
import time
import csv
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

EPSILON_DECAY = 40000
EPSILON_START = 0.8
EPSILON_END = 0.02

TARGET_UPDATE_FREQUENCY = 10 #每过多少局目标网络更新

env = MyEnv()
n_state = 32
n_action = env.action_space.n
LR_A = 0.0001
LR_C = 0.0002
GAMMA = 0.99
EPSILON = 0.1
A_update_steps = 10
C_update_steps = 10
batch = 32
num_ev = 10
count = int(num_ev / 10)
count_write = 0
write = np.zeros(count)
n_episode = 50000 * count



rewards = []
agent_DQN = Agent(n_input=n_state, n_output=n_action)
REWARD_BUFFER_DQN = np.empty(shape=n_episode)
write_r_DQN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s = env.reset()
s_DQN = s


for episode_i in range(n_episode):
    epsilon = np.interp(episode_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    random_sample = random.random()

    if random_sample <= epsilon:
        a = env.action_space.sample()
    else:
        a = agent_DQN.online_net.act(obs=s_DQN)

    s_next_DQN, r = env.step(act=a)
    agent_DQN.memo.add_memo(s_DQN, a, r, s_next_DQN)
    s_DQN = s_next_DQN
    REWARD_BUFFER_DQN[episode_i] = r

    # 从记忆池中抽取经验学习
    batch_s, batch_a, batch_r, batch_s_next = agent_DQN.memo.sample()

    # 计算targets
    target_q_values = agent_DQN.target_net(batch_s_next)  # 用目标网络获得下一状态的Q值
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    targets = batch_r + agent_DQN.GAMA * max_target_q_values

    # 计算q_values
    q_values = agent_DQN.online_net(batch_s)
    a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

    # 计算loss
    loss = nn.functional.smooth_l1_loss(targets, a_q_values)

    # 梯度下降
    agent_DQN.optimizer.zero_grad()
    loss.backward()
    agent_DQN.optimizer.step()

    write[episode_i % count] = r
    if (episode_i + 1) % count == 0:
        write_r_DQN[count_write] = np.sum(write)
        count_write = (count_write + 1) % 30

    if count_write == 0:
        rewards.append(np.mean(write_r_DQN))

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent_DQN.target_net.load_state_dict(agent_DQN.online_net.state_dict())

    if episode_i % 100 == 0:
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER_DQN[:episode_i])))
        print(a, r)








