import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ActorNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ActorNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        all_actions_prob = F.softmax(self.layer(x), dim=-1)
        return all_actions_prob


class CriticNet(nn.Module):
    def __init__(self, n_states):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        v = self.layer(x)
        return v


class PPO(nn.Module):
    def __init__(self, n_states, n_actions, lr_a, lr_c, gamma, epsilon, a_update_steps, c_update_steps):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.epsilon = epsilon
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps

        self._build()

    def _build(self):
        self.actor_model = ActorNet(self.n_states, self.n_actions)
        self.actor_old_model = ActorNet(self.n_states, self.n_actions)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr_a)

        self.critic_model = CriticNet(self.n_states)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr_c)

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        all_actions_prob = self.actor_model(s)
        dist = torch.distributions.Categorical(all_actions_prob)
        action = dist.sample()
        # print("prob: ", all_actions_prob)
        # action = torch.multinomial(all_actions_prob, 1)
        # print("action: ", action)
        # print(type(action))
        return action.detach().numpy()

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_)
        target = self.critic_model(s_).detach()  # torch.Size([1])
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)  # torch.Size([batch])

        return target_list

    def actor_learn(self, states, actions, advantage):  # states 和 actions 都是np.array类型
        # print("states: ", states)
        # print("actions: ", actions)
        # print("----------")
        states = torch.FloatTensor(states)

        actions = torch.cat([torch.as_tensor(a, dtype=torch.int64).unsqueeze(-1) for a in actions]).view(-1,
                                                                                                         1)  # to(torch.int64)

        # print("states: ", states)
        # print("actions: ", actions)
        # print(self.actor_model(states))

        select_action_prob_new = torch.gather(self.actor_model(states), dim=1, index=actions)

        select_action_prob_old = torch.gather(self.actor_old_model(states), dim=1, index=actions)

        ratio = select_action_prob_new / (select_action_prob_old + 1e-5)  # 添加一个小的常数以避免除以零的情况
        surr = ratio * advantage.reshape(-1, 1)  # torch.Size([batch, 1])
        loss = -torch.mean(
            torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states)  # torch.Size([batch, 1])
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach()  # torch.Size([batch])

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())  # 首先更新旧模型
        advantage = self.cal_adv(states, targets)  # 计算优势，也就是td-error

        for i in range(self.a_update_steps):  # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):  # 更新多次
            self.critic_learn(states, targets)
