import random

import numpy as np
import torch
import torch.nn as nn


class Replaymemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 64

        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        self.all_a = np.empty(shape=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_r = np.empty(shape=self.MEMORY_SIZE, dtype=np.float32)
        self.all_s_next = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self, s, a, r, s_next):
        self.all_s[self.t_memo] = s
        self.all_a[self.t_memo] = a
        self.all_r[self.t_memo] = r
        self.all_s_next[self.t_memo] = s_next
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.t_max >= self.BATCH_SIZE:
            idxes = random.sample(range(0, self.t_max), self.BATCH_SIZE)
        else:
            idxes = range(0, self.t_max)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_s_next = []

        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_s_next.append(self.all_s_next[idx])

        #将numpy转为torch
        batch_s_tensor = torch.as_tensor(np.array(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.array(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.array(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_s_next_tensor = torch.as_tensor(np.array(batch_s_next), dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_s_next_tensor


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=88),
            nn.Tanh(),
            nn.Linear(in_features=88, out_features=n_output)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_value = self(obs_tensor.unsqueeze(0))
        max_q_idx = torch.argmax(input=q_value)
        action = max_q_idx.detach().item()

        return action


class Agent:
    def __init__(self,n_input,n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMA = 0.99
        self.learning_rate = 0.001

        self.memo = Replaymemory(self.n_input, self.n_output)

        self.online_net = DQN(self.n_input, self.n_output)
        self.target_net = DQN(self.n_input, self.n_output)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)