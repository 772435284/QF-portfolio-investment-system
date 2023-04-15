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
        self.linear1 = nn.Linear((product_num + 1) * 1 * 32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.mean_linear = nn.Linear(64, product_num + 1)
        self.log_std_linear = nn.Linear(64, product_num + 1)

        self.log_std_min = -20
        self.log_std_max = 2
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)
    
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
        
        mean = self.mean_linear(fc2_out)
        log_std = self.log_std_linear(fc2_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal_distribution = torch.distributions.Normal(mean, std)
        x_t = normal_distribution.rsample()
        action = torch.tanh(x_t)
        log_prob = normal_distribution.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    

class Critic(nn.Module):
    def __init__(self, product_num, win_size, num_features):
        super(Critic, self).__init__()
        # First Q network
        self.conv1_q1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=(1, 3)
        )
        self.conv2_q1 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, win_size - 2)
        )
        self.linear1_q1 = nn.Linear((product_num + 1) * 1 * 32, 64)
        self.linear2_q1 = nn.Linear(64 + (product_num + 1), 64)
        self.linear3_q1 = nn.Linear(64, 1)

        # Second Q network
        self.conv1_q2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=(1, 3)
        )
        self.conv2_q2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, win_size - 2)
        )
        self.linear1_q2 = nn.Linear((product_num + 1) * 1 * 32, 64)
        self.linear2_q2 = nn.Linear(64 + (product_num + 1), 64)
        self.linear3_q2 = nn.Linear(64, 1)

    def reset_parameters(self):
        hidden_init_list = [self.linear1_q1, self.linear2_q1, self.linear1_q2, self.linear2_q2]
        for layer in hidden_init_list:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.linear3_q1.weight.data.uniform_(-3e-3, 3e-3)
        self.linear3_q2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # First Q network
        x1 = F.relu(self.conv1_q1(state))
        x1 = F.relu(self.conv2_q1(x1))
        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.linear1_q1(x1))
        x1 = torch.cat([x1, action], dim=1)
        x1 = F.relu(self.linear2_q1(x1))
        q1 = self.linear3_q1(x1)

        # Second Q network
        x2 = F.relu(self.conv1_q2(state))
        x2 = F.relu(self.conv2_q2(x2))
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.linear1_q2(x2))
        x2 = torch.cat([x2, action], dim=1)
        x2 = F.relu(self.linear2_q2(x2))
        q2 = self.linear3_q2(x2)

        return q1, q2

        

class SAC:
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
        self.alpha = 0.01
        self.gamma = config.gamma
        self.tau = config.tau
        self.num_features = len(config.factor)
        
        # Initialize Actor and Critic networks
        self.actor = Actor(product_num, window_size, self.num_features).to(self.device)
        self.actor_target = Actor(product_num, window_size, self.num_features).to(self.device)
        self.critic = Critic(product_num, window_size, self.num_features).to(self.device)
        self.critic_target = Critic(product_num, window_size, self.num_features).to(self.device)

        # Set target networks' weights equal to their corresponding networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)

        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.baseline_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)

        
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy().flatten()
    

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # Convert batches to tensors and move them to the device
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Update Critic
        # Calculate target Q-values using the target networks
        with torch.no_grad():
            next_action_batch, next_action_log_prob_batch = self.actor_target.sample(next_state_batch)
            q1_target_batch, q2_target_batch = self.critic_target(next_state_batch, next_action_batch)
            min_q_target_batch = torch.min(q1_target_batch, q2_target_batch)
            target_q_batch = reward_batch + (1 - done_batch) * self.gamma * (min_q_target_batch - self.alpha * next_action_log_prob_batch)
        
         # Calculate current Q-values
        q1_batch, q2_batch = self.critic(state_batch, action_batch)

        # Compute Critic loss
        critic_loss = F.mse_loss(q1_batch, target_q_batch) + F.mse_loss(q2_batch, target_q_batch)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        # Calculate Actor loss
        actions_pred, actions_pred_log_prob = self.actor.sample(state_batch)
        q1_pred_batch, q2_pred_batch = self.critic(state_batch, actions_pred)
        min_q_pred_batch = torch.min(q1_pred_batch, q2_pred_batch)
        actor_loss = torch.mean(self.alpha * actions_pred_log_prob - min_q_pred_batch)

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update Actor Target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




    def train(self):
        num_episode = self.config.episode
        batch_size = self.config.batch_size
        
        self.buffer = ReplayBuffer(self.config.buffer_size)
        
        creator = obs_creator(self.config.norm_method,self.config.norm_type)

        for i in range(num_episode):
            previous_observation, _ = self.env.reset()
            previous_observation = creator.create_obs(previous_observation)
            done = False

            ep_reward = 0
            for j in range (self.config.max_step):
                action = self.select_action(previous_observation)
                observation, reward, done, _ = self.env.step(action)
                observation = creator.create_obs(observation)
                self.buffer.add(previous_observation, action, reward, observation, done)
                previous_observation =  observation

                if self.buffer.size() >= batch_size:
                    experiences = self.buffer.sample_batch(batch_size)
                    self.update(*experiences)
                ep_reward += reward
                if done or j == self.config.max_step - 1:
                    print('Episode: {:d}, Reward: {:.2f}'.format(i, ep_reward))
                    break
        print('Finish.')
        torch.save(self.actor.state_dict(), path_join(self.config.baseline_dir, self.current_agent.name))