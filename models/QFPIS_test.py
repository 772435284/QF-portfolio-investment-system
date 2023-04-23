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
from environment.QF_env import envs

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
        self.linear3 = nn.Linear(64, product_num+1)
    
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
        fc3_out = torch.tanh(fc3_out)  # Replace Softmax with Tanh

        # Normalize the output to ensure the sum of the holding ratios is 1
        fc3_out_normalized = F.normalize(fc3_out, p=1, dim=1)
        
        return fc3_out_normalized
    
class Critic(nn.Module):
    def __init__(self, product_num, win_size, num_features, action_size):
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
        self.linear1 = nn.Linear((product_num + 1) * 1 * 32 + action_size, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, state, action):
        conv1_out = self.conv1(state)
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        # Flatten
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        # Concatenate flattened state and action
        concat_input = torch.cat((conv2_out, action), dim=1)
        fc1_out = self.linear1(concat_input)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.linear2(fc1_out)
        
        return fc2_out
    

# Define Policy network
class Policy(nn.Module):
    def __init__(self,product_num, win_size,num_features,action_size):
        super(Policy, self).__init__()

        self.lstm = nn.LSTM(win_size,32,2)

        self.linear1 = nn.Linear((product_num+1)*num_features*1*32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64,action_size)

        # Define the  vars for recording log prob and reawrd
        self.saved_log_probs = []
        self.rewards = []
        self.product_num = product_num
        self.num_features = num_features
        self.win_size = win_size

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):

        state = torch.reshape(state, (-1, 1, self.win_size))
        #print(state)
        lstm_out, _ = self.lstm(state)
        #print(lstm_out)
        batch_n,win_s,hidden_s = lstm_out.shape
        lstm_out = lstm_out.view(batch_n, win_s*hidden_s)
        lstm_out = torch.reshape(lstm_out, (-1, (self.product_num+1)*self.num_features, 32))
        lstm_out = lstm_out.view(lstm_out.size(0), -1)
        fc1_out = self.linear1(lstm_out)
        #fc1_out = F.relu(fc1_out)
        fc2_out = self.linear2(fc1_out)
        #fc2_out = F.relu(fc2_out)
        fc3_out = self.linear3(fc2_out)
        fc3_out = F.softmax(fc3_out,dim=1)

        return fc3_out


# Define Critic network
class Critic(nn.Module):
    def __init__(self, product_num, win_size,num_features):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels =  num_features,
            out_channels = 32,
            kernel_size = (1,3),
            #stride = (1,3)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (1, win_size-2),
            #stride = (1, win_size-2)
        )
        self.linear1 = nn.Linear((product_num + 1)*1*32, 64)
        self.linear2 = nn.Linear((product_num + 1), 64)
        self.linear3 = nn.Linear(64, 1)
    
    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        # Observation channel
        conv1_out = self.conv1(state)
        conv1_out = F.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.relu(conv2_out)
        # Flatten
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        fc1_out = self.linear1(conv2_out)
        # Action channel
        fc2_out = self.linear2(action)
        obs_plus_ac = torch.add(fc1_out,fc2_out)
        obs_plus_ac = F.relu(obs_plus_ac)
        fc3_out = self.linear3(obs_plus_ac)
        
        return fc3_out

class QFPIS(object):
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
        self.action_size = config.qpl_level + 1

        self.actor = Actor(product_num, window_size, self.num_features).to(device)
        self.actor_target = Actor(product_num, window_size, self.num_features).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)

        self.critic = Critic(product_num, window_size, self.num_features).to(device)
        self.critic_target = Critic(product_num, window_size, self.num_features).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)

        self.policy = Policy(product_num, window_size, self.num_features,self.action_size).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=1e-4)

        self._initialize_target_networks()

        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.ddpg_model_dir, exist_ok=True)
        os.makedirs(config.pga_model_dir, exist_ok=True)
        os.makedirs(config.baseline_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)

    def _initialize_target_networks(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state).squeeze(0).cpu().numpy()
        self.actor.train()
        return action
    
    # Here is the code for the policy gradient actor
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        # Get the probability distribution
        #print(state)
        probs = self.policy(state)
        #print(probs)
        m = Categorical(probs)
        # Sample action from the distribution
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def policy_learn(self, eps):
        R = 0
        policy_loss = []
        returns = []

         # Reversed Traversal and calculate cumulative rewards for t to T
        for r in self.policy.rewards[::-1]:
            R = r + 0.95 * R # R: culumative rewards for t to T
            returns.insert(0, R) # Evaluate the R and keep original order

        returns = torch.tensor(returns).to(self.device)
        # Normalized returns
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # After one episode, update once
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            # Actual loss definition:
            policy_loss.append(-log_prob * R)
        self.policy_optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy_optim.step()

        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

        return policy_loss

    
    def update_networks(self, state_batch, action_batch, reward_batch,done_batch, next_state_batch ):
        states = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        actions = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
        dones = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Update Critic
        next_actions = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, next_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def validation(self):
        self.config.mode = "Val"
        self.env =  envs(self.config)
        self.actor.eval()
        self.policy.eval()
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        eps = 1e-8
        actions = []
        weights = []
        observation, info = self.env.reset()
        observation = creator.create_obs(observation)
        done = False
        ep_reward = 0
        while not done:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(observation).squeeze(0).cpu().numpy()
                actions_prob = self.policy(observation)
            m = Categorical(actions_prob)
            # Selection action by sampling the action prob
            action_policy = m.sample()
            observation, reward,policy_reward, done, info = self.env.step(action,action_policy)
            ep_reward += reward
            r = info['log_return']
            observation =  creator.create_obs(observation)
        SR, MDD, FPV, CRR, AR, AV = self.env.render()
        self.config.mode = "Train"
        self.env =  envs(self.config)
        self.actor.train()
        self.policy.train()
        return FPV

    def train(self):
        num_episode = self.config.episode
        batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        eps = np.finfo(np.float32).eps.item()
        self.buffer = ReplayBuffer(self.config.buffer_size)
        total_step = 0
        writer = SummaryWriter(logdir=self.summary_path)
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        stop_tolerance = 0
        last_fpv = float('-inf')
        all_fpv = [float('-inf')]

        for i in range(num_episode):
            previous_observation, _ = self.env.reset()
            previous_observation = creator.create_obs(previous_observation)
            ep_reward  = 0

            for j in range(self.config.max_step):
                action = self.select_action(previous_observation)
                action_policy = self.act(previous_observation)
                observation, reward, policy_reward, done, _ = self.env.step(action,action_policy)
                observation = creator.create_obs(observation)
                ep_reward += reward
                self.policy.rewards.append(policy_reward)

                self.buffer.add(previous_observation, action, reward, done, observation)
                previous_observation =  observation

                if self.buffer.size() >= batch_size:
                    
                    experiences = self.buffer.sample_batch(batch_size)
                    self.update_networks(*experiences)

                if done or j == self.config.max_step - 1:
                    print('Episode: {:d}, Reward: {:.2f}'.format(i, ep_reward))
                    break
            policy_loss = self.policy_learn(eps)
            fpv = self.validation()
            if last_fpv <  fpv:
                stop_tolerance = 0
            else:
                stop_tolerance += 1
            # Save the best model:
            if fpv > max(all_fpv):
                torch.save(self.actor.state_dict(), path_join(self.config.baseline_dir, f'{self.current_agent.name}_QPL_{self.config.qpl_level}_{self.config.data_dir}'))
                print("Best Model saved !!!")
            print("FPV:",fpv)
            print("last_FPV:",last_fpv)
            print("max",max(all_fpv))
            print(stop_tolerance)
            last_fpv = fpv
            all_fpv.append(fpv)
            if stop_tolerance >= self.config.tolerance:
                break
        print('Finish.')               
