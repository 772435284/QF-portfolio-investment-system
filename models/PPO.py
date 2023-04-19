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
    def __init__(self, product_num, win_size, num_features, action_std_init=0.5):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels =  num_features,
            out_channels = 32,
            kernel_size = (1,3),
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (1, win_size-2),
        )
        self.linear1 = nn.Linear((product_num + 1)*1*32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3_mean = nn.Linear(64, product_num + 1)
        self.linear3_log_std = nn.Linear(64, product_num + 1)

        self.action_std = torch.full((product_num,), action_std_init)

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3_mean.weight.data.uniform_(-3e-3, 3e-3)
        self.linear3_log_std.weight.data.uniform_(-3e-3, 3e-3)

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
        
        mean = self.linear3_mean(fc2_out)
        log_std = self.linear3_log_std(fc2_out)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        action_log_prob = normal.log_prob(action)
        return action, action_log_prob.sum(dim=-1).unsqueeze(-1)


class Critic(nn.Module):
    def __init__(self, product_num, win_size, num_features):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels =  num_features,
            out_channels = 32,
            kernel_size = (1,3),
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (1, win_size-2),
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
    
class PPO:
    config: GlobalConfig
    actor_noise: Callable
    summary_path: str = path_join('train_results', 'ddpg')
    current_agent: AgentProps
    price_history: pd.DataFrame
    trading_dates: List[str]
    window_size: int
    use_cuda: bool

    def __init__(self, env, window_size: int, actor_noise: Callable, device: str, config: GlobalConfig):
        self.device = device
        
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
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)

        self.summary_path = path_join(config.summary_path, config.model_name)

        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.baseline_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)


    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, action_log_prob = self.actor.get_action(state)
        action = np.squeeze(action)
        return action.detach().cpu().numpy(), action_log_prob.detach().cpu().numpy()
    
    def update(self, state, action, action_log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        action_log_prob = torch.FloatTensor(action_log_prob).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)

        # Calculate the state values
        state_value = self.critic(state)
        next_state_value = self.critic(next_state)

        # Calculate the advantage
        target_value = reward + (1 - done) * next_state_value
        advantage = target_value - state_value

        # Update the critic network
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate the new action log probabilities and the ratio
        _, new_action_log_prob = self.actor.get_action(state)
        ratio = (new_action_log_prob - action_log_prob).exp()

        # Calculate the actor loss using the clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        actor_loss = -torch.min(ratio * advantage.detach(), clipped_ratio * advantage.detach()).mean()
        
        # Update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self):
        num_episode = self.config.episode
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        for i in range(num_episode):
            previous_observation, _ = self.env.reset()
            previous_observation = creator.create_obs(previous_observation)
            episode_reward = 0
            for j in range(self.config.max_step):
                action, action_log_prob = self.get_action(previous_observation)
                observation, reward, done, _ = self.env.step(action)
                observation = creator.create_obs(observation)

                # Train the actor and critic networks using the collected data
                self.update(previous_observation, action, action_log_prob, reward, observation, done)
                
                previous_observation =  observation
                episode_reward += reward
                if done or j == self.config.max_step - 1:
                    print('Episode: {:d}, Reward: {:.2f}'.format(i, episode_reward))
                    break
        print('Finish.')
        torch.save(self.actor.state_dict(), path_join(self.config.baseline_dir, f'{self.current_agent.name}_{self.config.data_dir}'))
