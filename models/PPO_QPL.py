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


class PPO_QPL:
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
        self.action_size = config.qpl_level + 1

        self.actor = Actor(product_num, window_size, self.num_features).to(self.device)
        self.critic = Critic(product_num, window_size, self.num_features).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_learning_rate)

        self.policy = Policy(product_num, window_size, self.num_features,self.action_size).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.config.policy_learning_rate)        

        self.summary_path = path_join(config.summary_path, config.model_name)

        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.baseline_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)
        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.ddpg_model_dir, exist_ok=True)
        os.makedirs(config.pga_model_dir, exist_ok=True)
        os.makedirs(config.baseline_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)


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

    def validation(self):
        self.config.mode = "Val"
        self.env =  envs(self.config)
        self.actor.eval()
        self.policy.eval()
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        observation, info = self.env.reset()
        observation = creator.create_obs(observation)
        done = False
        ep_reward = 0
        actions = []
        plot_action = [[0.0]*10]
        plot_action = np.array(plot_action)
        i = 0
        while not done:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, action_log_prob = self.actor.get_action(observation)
                action = np.squeeze(action)
                action = action.detach().cpu().numpy()
                actions_prob = self.policy(observation)
            m = Categorical(actions_prob)
            # Selection action by sampling the action prob
            action_policy = m.sample()
            actions.append(action_policy.cpu().numpy())
            if i == 9:
                plot_action = action
            observation, reward,policy_reward, done, info = self.env.step(action,action_policy)
            ep_reward += reward
            r = info['log_return']
            observation =  creator.create_obs(observation)
            i += 1
        SR, MDD, FPV, CRR, AR, AV = self.env.render()
        self.config.mode = "Train"
        self.env =  envs(self.config)
        self.actor.train()
        self.policy.train()
        return FPV, plot_action, actions


    def train(self):
        eps = np.finfo(np.float32).eps.item()
        num_episode = self.config.episode
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        stop_tolerance = 0
        last_fpv = float('-inf')
        all_fpv = [float('-inf')]
        all_plot_action = []
        for i in range(num_episode):
            plot_policy_action = []
            previous_observation, _ = self.env.reset()
            previous_observation = creator.create_obs(previous_observation)
            episode_reward = 0
            for j in range(self.config.max_step):
                action, action_log_prob = self.get_action(previous_observation)
                action_policy = self.act(previous_observation)
                if (i+1)  % 3 == 0:
                    plot_policy_action.append(action_policy)
                observation, reward, policy_reward, done, _ = self.env.step(action,action_policy)
                observation = creator.create_obs(observation)
                self.policy.rewards.append(policy_reward)
                # Train the actor and critic networks using the collected data
                self.update(previous_observation, action, action_log_prob, reward, observation, done)
                
                previous_observation =  observation
                episode_reward += reward
                if done or j == self.config.max_step - 1:
                    print('Episode: {:d}, Reward: {:.2f}'.format(i, episode_reward))
                    break
            policy_loss = self.policy_learn(eps)
            plot_policy_action = np.array(plot_policy_action)
            if (i+1)  % 3 == 0:
                pd.DataFrame(plot_policy_action).to_csv('backtest_result/train_policy_actions/'+f"{self.current_agent.name}_train_QPL_{self.config.qpl_level}_ep_{i+1}_{self.config.data_dir}_action_record.csv", index=None)
            fpv, plot_action,val_policy_action = self.validation()
            all_plot_action.append(plot_action)
            if (i+1)  % 3 == 0:
                pd.DataFrame(val_policy_action).to_csv('backtest_result/val_policy_actions/'+f"{self.current_agent.name}_val_QPL_{self.config.qpl_level}_ep_{i+1}_{self.config.data_dir}_action_record.csv", index=None)
            if last_fpv <  fpv:
                stop_tolerance = 0
            else:
                stop_tolerance += 1
            # Save the best model:
            if fpv > max(all_fpv):
                torch.save(self.actor.state_dict(), path_join(self.config.ddpg_model_dir, f'{self.current_agent.name}_QPL_{self.config.qpl_level}_{self.config.data_dir}'))
                torch.save(self.policy.state_dict(), path_join(self.config.pga_model_dir, f'{self.current_agent.name}_QPL_{self.config.qpl_level}_{self.config.data_dir}'))
                print("Best Model saved !!!")
            print("FPV:",fpv)
            print("last_FPV:",last_fpv)
            print("max",max(all_fpv))
            print(stop_tolerance)
            last_fpv = fpv
            all_fpv.append(fpv)
            if stop_tolerance >= self.config.tolerance:
                break
        all_fpv = np.array(all_fpv)
        all_plot_action = np.array(all_plot_action)
        np.save(f'backtest_result/{self.current_agent.name}_val_record_QPL_{self.config.qpl_level}_{self.config.data_dir}', all_fpv)
        np.save(f'backtest_result/{self.current_agent.name}_plot_action_QPL_{self.config.qpl_level}_{self.config.data_dir}', all_plot_action)
        print('Finish.')
        print('Finish.')
        #torch.save(self.actor.state_dict(), path_join(self.config.baseline_dir, f'{self.current_agent.name}_QPL_{self.config.qpl_level}_{self.config.data_dir}'))
