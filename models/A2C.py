import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
from torch.autograd import Variable
import pandas as pd
import numpy as np
from tools.ddpg.replay_buffer import ReplayBuffer
from api_types import GlobalConfig, AgentProps
from typing import Callable, List, cast, OrderedDict
from os.path import join as path_join
from torch.distributions import Categorical
from utils.utils import hidden_init
from tensorboardX import SummaryWriter
from observation.obs_creator import obs_creator


class Actor(nn.Module):
    def __init__(self, product_num, win_size, num_features):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=(1, 3),
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, win_size - 2),
        )
        self.linear1 = nn.Linear((product_num + 1)*1*32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, product_num + 1)
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        conv1_out = self.conv1(state)
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        # Flatten
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        fc1_out = self.linear1(conv2_out)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.linear2(fc1_out)
        fc2_out = F.relu(fc2_out)
        fc3_out = self.linear3(fc2_out)
        fc3_out = F.softmax(fc3_out, dim=1)
        
        return fc3_out
    

class Critic(nn.Module):
    def __init__(self, product_num, win_size, num_features):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=(1, 3),
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, win_size - 2),
        )
        self.linear1 = nn.Linear((product_num + 1) * 1 * 32, 64)
        self.linear2 = nn.Linear(64, 1)

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        conv1_out = self.conv1(state)
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        # Flatten
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        fc1_out = self.linear1(conv2_out)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.linear2(fc1_out)
        
        return fc2_out


class A2C:
    config: GlobalConfig
    actor_noise: Callable
    summary_path: str = path_join('train_results', 'ddpg')
    current_agent: AgentProps
    price_history: pd.DataFrame
    trading_dates: List[str]
    window_size: int
    use_cuda: bool

    def __init__(self, env, window_size: int, actor_noise: Callable, device: str, config: GlobalConfig):
        # Load configuration
        self.config = config
        assert self.config is not None, "Can't load config"
        
        assert type(config.use_agents) == int, "You must specify one agent for training!"
        agent_index = cast(int, config.use_agents)
        self.current_agent = AgentProps(config.agent_list[agent_index])
        product_num = len(self.current_agent.product_list)
        self.use_cuda = config.use_cuda and cuda.is_available()
        self.window_size = window_size
        self.actor_noise = actor_noise
        self.env = env
        self.device = device
        self.num_features = len(config.factor)
        
        self.actor = Actor(product_num, window_size, self.num_features).to(self.device)
        self.critic = Critic(product_num, window_size, self.num_features).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = self.config.actor_learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = self.config.critic_learning_rate)
        self.gamma = config.gamma

        self.summary_path = path_join(config.summary_path, config.model_name)

        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.ddpg_model_dir, exist_ok=True)
        os.makedirs(config.pga_model_dir, exist_ok=True)
        os.makedirs(config.baseline_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
        action_probs = action_probs.cpu().numpy().squeeze()
        return action_probs

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Critic loss
        state_value = self.critic(state)
        next_state_value = self.critic(next_state).detach()
        target_value = reward + (1 - done) * self.gamma * next_state_value
        critic_loss = F.mse_loss(state_value, target_value)

        # Actor loss
        action_probs = self.actor(state)
        state_value_estimate = self.critic(state).detach()
        advantage = target_value - state_value_estimate
        actor_loss = -torch.sum(torch.log(action_probs) * action * advantage)

        # Update networks
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.item(), critic_loss.item()

    def train(self):
        episode_rewards = []
        num_episode = self.config.episode
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        total_step = 0
        for i in range(num_episode):
            previous_observation, _ = self.env.reset()
            previous_observation = creator.create_obs(previous_observation)

            episode_reward = 0
            

            for j in range(self.config.max_step):
                action = self.select_action(previous_observation)
                observation, reward, done, _ = self.env.step(action)
                observation = creator.create_obs(observation)
                
                actor_loss, critic_loss = self.update(previous_observation, action, reward, observation, done)
                previous_observation =  observation

                
                episode_reward += reward
                if done or j == self.config.max_step - 1:
                    print('Episode: {:d}, Reward: {:.2f}'.format(i, episode_reward))
                    break
            episode_rewards.append(episode_reward)

        print('Finish.')
        torch.save(self.actor.state_dict(), path_join(self.config.baseline_dir, f'{self.current_agent.name}_{self.config.data_dir}'))
        return episode_rewards
    
    