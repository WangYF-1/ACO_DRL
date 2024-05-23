import random
from random import *

import gym
import numpy as np
from gym import spaces
import json

class MyEnv(gym.Env):
    def __init__(self):
        self.n_customer = 5 #消费者个数
        self.n_producer = 5 #生产者个数
        #with open('ev.json', 'r') as file:
            #ev = json.load(file)
        #self.ev = ev
        #self.ev_count = 0
        self.action_space = spaces.Discrete(2520)
        self.lamada_s = 0.7
        self.lamada_b = 1.2
        self.lamada_v2v = 0.9
        self.observation_space = [spaces.Box(low=9,high=21,shape=(1, self.n_customer + self.n_producer),dtype=np.uint8),
                                  spaces.Box(low=89,high=101,shape=(1, self.n_customer + self.n_producer),dtype=np.uint8),
                                  spaces.Box(low=0,high=50,shape=(1, self.n_customer + self.n_producer), dtype=np.float32),
                                  spaces.Box(low=0,high=50,shape=(1, self.n_customer + self.n_producer), dtype=np.float32),
                                  spaces.Box(low=-1,high=2,shape=(1, self.n_customer +self.n_producer),dtype=np.uint8)]
        self.state = None
        self.customer_init = []
        self.customer_exp = []
        self.producer_init = []
        self.producer_exp = []
        self.action_dic = []
        self.action_dic_write()


    def action_dic_write(self):
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for m in range(5):
                        for n in range(5):
                            if i != j and i != k and i != m and i != n and j != k and j != m and j != n and k != m and k != n and m != n:
                                for s1 in range(21):
                                    self.action_dic.append(
                                        [i, j, k, m, n, 70 + s1,  70 + s1, 70 + s1, 70 + s1, 70 + s1])

    def step(self, act):
        #动作映射
        action = self.action_dic[act]
        #计算奖励
        reward = self.get_reward_allv2v(action)
        next_observation = self.reset()
        return next_observation, reward

    def get_reward_allv2v(self, action):
        match = []
        match_energy = []
        #惩罚值
        sita1 = 1.5
        sita2 = 1.5
        sita3 = 1.5
        sita4 = 1.5
        for i in range(len(action)):
            if i < self.n_customer:
                match.append(action[i])
            else:
                match_energy.append(action[i])
        reward = 0
        for i in range(self.n_customer):

            pm = 0
            pn = 0
            if self.customer_init[i] + match_energy[i] <= self.customer_exp[i]:
                rm = -self.lamada_v2v * match_energy[i]
                pm = sita1 * (self.customer_exp[i] - (self.customer_init[i] + match_energy[i]))
            if self.customer_init[i] + match_energy[i] > self.customer_exp[i]:
                rm = -self.lamada_v2v * (self.customer_exp[i] - self.customer_init[i])
                pm = sita2 * (self.customer_init[i] + match_energy[i] - self.customer_exp[i])

            if self.producer_init[match[i]] - match_energy[i] < self.producer_exp[match[i]]:
                rn = self.lamada_v2v * match_energy[i] - 2
                pn = sita3 * (self.producer_exp[match[i]] - (self.producer_init[match[i]] - match_energy[i]))
            if self.producer_init[match[i]] - match_energy[i] >= self.producer_exp[match[i]]:
                rn = self.lamada_v2v * match_energy[i] - 2
                pn = sita4 * (self.producer_init[match[i]] - match_energy[i] - self.producer_exp[match[i]])
            reward = reward + rm * 0.3 + rn * 0.7 - pm - pn

        return reward





    def reset(self):
        #e_init = self.ev[self.ev_count][0]
        #e_exp = self.ev[self.ev_count][1]
        #self.ev_count += 1
        e_init = []
        e_exp = []
        lamada_s_array = [self.lamada_s]
        lamada_b_array = [self.lamada_b]
        di = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        s = []

        self.customer_init = []
        self.customer_exp = []
        self.producer_init = []
        self.producer_exp = []

        for i in range(self.n_customer):
            e_init.append(randint(10, 20))
            self.customer_init.append(e_init[i])
            e_exp.append(randint(90, 100))
            self.customer_exp.append(e_exp[i])

        for i in range(self.n_producer):
            e_init.append(randint(90, 100))
            self.producer_init.append(e_init[i + 5])
            e_exp.append(randint(10, 20))
            self.producer_exp.append(e_exp[i + 5])

        s.extend(e_init)
        s.extend(e_exp)
        s.extend(lamada_s_array)
        s.extend(lamada_b_array)
        s.extend(di)

        self.state = np.array(s)

        return self.state