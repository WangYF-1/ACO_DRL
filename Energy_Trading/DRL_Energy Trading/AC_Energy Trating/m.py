import torch
import random
from ag import Actor_Critic
from Env import MyEnv
from painter import painter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
EPSILON_DECAY = 40000
EPSILON_START = 0.8
EPSILON_END = 0.02
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

env = MyEnv()
n_state = 32
n_action = env.action_space.n
painter = painter()
rewards = []

num_ev = 10
count = int(num_ev / 10)
count_write = 0
write = np.zeros(count)
n_episode = 50000 * count
start_time = time.time()
agent = Actor_Critic(n_action, n_state)
s = env.reset()
REWARD_BUFFER = np.empty(shape=n_episode)
write_r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for episode_i in range(n_episode):
    epsilon = np.interp(episode_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    random_sample = random.random()

    if random_sample <= epsilon:
        a = env.action_space.sample()
        a = torch.tensor(a)
        a = a.detach().numpy()
    else:
        a, log_prob = agent.get_action(s)
    s_next, r = env.step(act=a)
    REWARD_BUFFER[episode_i] = r

    if episode_i % 8 == 0:
        agent.learn(a, s, s_next, r)
    s = s_next
    write[episode_i % count] = r
    if (episode_i + 1) % count == 0:
        write_r[count_write] = np.sum(write)
        count_write = (count_write + 1) % 30

    if count_write == 0:
        rewards.append(np.mean(write_r))

    if episode_i % 100 == 0:
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))
        print(a, r)


