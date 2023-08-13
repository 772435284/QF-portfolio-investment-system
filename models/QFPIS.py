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
import math
from torch.optim.lr_scheduler import _LRScheduler



class CosineScheduleWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, num_cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        super(CosineScheduleWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return [base_lr * (1 + math.cos(math.pi * self.num_cycles * progress)) / 2
                    for base_lr in self.base_lrs]

# Define actor network
class Actor(nn.Module):
    def __init__(self,product_num, win_size,num_features):
        super(Actor, self).__init__()
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
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64,product_num + 1)
    
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
        fc3_out = torch.tanh(fc3_out)  # Replace Softmax with Tanh

        # Normalize the output to ensure the sum of the holding ratios is 1
        fc3_out_normalized = F.normalize(fc3_out, p=1, dim=1)
        
        return fc3_out_normalized

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


class QFPIS(object):
    
    config: GlobalConfig
    actor_noise: Callable
    summary_path: str
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
        self.action_size = config.qpl_level + 1
        self.num_features = len(config.factor)
        # self.price_history = price_history
        # self.trading_dates = trading_dates
        # assert len(trading_dates) == config.max_step+self.window_size
        
        self.summary_path = path_join(config.summary_path, config.model_name)
        
        self.actor = Actor(product_num,window_size,self.num_features).to(self.device)
        self.actor_target = Actor(product_num,window_size,self.num_features).to(self.device)
        self.critic = Critic(product_num,window_size,self.num_features).to(self.device)
        self.critic_target = Critic(product_num,window_size, self.num_features).to(self.device)
        

        # Here is the code for the policy-gradeint
        
        self.policy = Policy(product_num, window_size, self.num_features,self.action_size).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.config.policy_learning_rate)
        
        self.actor.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_target.reset_parameters()
        self.actor.reset_parameters()
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.config.actor_learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.config.critic_learning_rate)
        
        self.actor_scheduler = CosineScheduleWithWarmup(self.actor_optim, warmup_steps=10, total_steps=self.config.episode)
        self.critic_scheduler = CosineScheduleWithWarmup(self.critic_optim, warmup_steps=10, total_steps=self.config.episode)
        self.policy_scheduler = CosineScheduleWithWarmup(self.policy_optim, warmup_steps=10, total_steps=self.config.episode)

        self.actor_target.load_state_dict(cast(OrderedDict[str, torch.Tensor], self.actor.state_dict()))
        self.critic_target.load_state_dict(cast(OrderedDict[str, torch.Tensor], self.critic.state_dict()))
    
        os.makedirs(config.train_intermediate_dir, exist_ok=True)
        os.makedirs(config.ddpg_model_dir, exist_ok=True)
        os.makedirs(config.pga_model_dir, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)



    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).squeeze(0).cpu().detach().numpy()+ self.actor_noise()
        return action
    
    def critic_learn(self, state, action, predicted_q_value):
        actual_q = self.critic(state, action)
        target_Q = torch.tensor(predicted_q_value, dtype=torch.float).to(self.device)
        target_Q = target_Q.reshape(-1, 1)
        target_Q=Variable(target_Q,requires_grad=True)
        td_error  = F.mse_loss(actual_q, target_Q)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()
        self.critic_scheduler.step()
        return predicted_q_value,td_error
    
    def actor_learn(self, state):
        loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.actor_scheduler.step()
        return loss
    
    def soft_update(self, net_target, net, tau):
        for target_param, param  in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    # Here is the code for the policy gradient actor
    def select_action(self, state):
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

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
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
        self.policy_scheduler.step()

        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

        return policy_loss

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
        wealth = self.config.wealth
        i = 0
        plot_action = []
        while not done:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(observation).squeeze(0).cpu().detach().numpy()
                actions_prob = self.policy(observation)
            m = Categorical(actions_prob)
            # Selection action by sampling the action prob
            action_policy = m.sample()
            actions.append(action_policy.cpu().numpy())
            #print(action)
            plot_action.append(action)
            w1 = np.clip(action, 0, 1)  # np.array([cash_bias] + list(action))  # [w0, w1...]
            w1 /= (w1.sum() + eps)
            weights.append(w1)
            observation, reward,policy_reward, done, info = self.env.step(action,action_policy)
            ep_reward += reward
            observation = creator.create_obs(observation)
            i += 1
        SR, MDD, FPV, CRR, AR, AV = self.env.render()
        self.config.mode = "Train"
        self.env =  envs(self.config)
        self.actor.train()
        self.policy.train()
        plot_action = np.array(plot_action)
        return FPV, plot_action, actions




    def train(self):
        num_episode = self.config.episode
        batch_size = self.config.batch_size
        gamma = self.config.gamma
        tau = self.config.tau
        eps = np.finfo(np.float32).eps.item()
        self.buffer = ReplayBuffer(self.config.buffer_size)
        total_step = 0
        moving_average_reward = 0
        writer = SummaryWriter(self.summary_path)
        creator = obs_creator(self.config.norm_method,self.config.norm_type)
        stop_tolerance = 0
        last_fpv = float('-inf')
        all_fpv = [float('-inf')]
        all_plot_action = []
        # Main training loop
        for i in range(num_episode):
            plot_policy_action = []
            previous_observation, _ = self.env.reset()
            # Normalization
            previous_observation = creator.create_obs(previous_observation)
            #previous_observation = np.expand_dims(previous_observation, axis=0)
            ep_reward = 0
            ep_ave_max_q = 0
            
            # Keep sampling until done
            for j in range (self.config.max_step):
                # ================================================
        		# 1. Given state st, take action at based on actor
        		# ================================================
                
                action = self.act(previous_observation)
                
                

                action_policy = self.select_action(previous_observation)
                if (i+1)  % 3 == 0:
                    plot_policy_action.append(action_policy)
                #print(action_policy)
                # ================================================
        		# 2. Obtain reward rt and reach new state st+1
                # ================================================
                observation_origin, reward, policy_reward, done, _ = self.env.step(action,action_policy)
                
                # ================================================
                # For Policy Gradient, append reward
                # ================================================
                self.policy.rewards.append(policy_reward)
                # ================================================
                # For Policy Gradient, Update network parameter
                # ================================================
                
                
                observation = creator.create_obs(observation_origin)
                #observation = np.expand_dims(observation, axis=0)
                # ================================================
        		# 3. Store (st, at, rt, st+1)
        		# ================================================
                self.buffer.add(previous_observation, action, reward, done, observation)
                if self.buffer.size() >= batch_size:
                    # ==========================================
        			# 4. Sample (si,ai,ri,si+1) from the buffer
        			# ==========================================
                    s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    # Convert to torch tensor
                    s_batch = torch.tensor(s_batch, dtype=torch.float).to(self.device)
                    a_batch = torch.tensor(a_batch, dtype=torch.float).to(self.device)
                    r_batch = torch.tensor(r_batch, dtype=torch.float).to(self.device)
                    t_batch = torch.tensor(t_batch, dtype=torch.float).to(self.device)
                    s2_batch = torch.tensor(s2_batch, dtype=torch.float).to(self.device)
                    target_q = self.critic_target(s2_batch,self.actor_target(s2_batch)).cpu().detach()
                    y_i = []
                    for k in range(batch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k].cpu().numpy() + gamma * target_q[k].numpy())
                    #y_i = r_batch + gamma * target_q
                    # =========================================================
        			# 6. Update the parameters of Q to make Q(si,ai) close to y
        			# =========================================================
                    predicted_q_value,td_error = self.critic_learn(s_batch, a_batch,np.reshape(y_i, (batch_size, 1)))
                    writer.add_scalar('TD error', td_error, global_step=total_step)
                    ep_ave_max_q += np.amax(predicted_q_value)
                    
                    # ================================================================
        			# 7. Update the parameters of of actor to maximize Q(si,actor(si))
        			# ================================================================
                    actor_loss = self.actor_learn(s_batch)
                    writer.add_scalar('Actor loss', actor_loss, global_step=total_step)
                    # ===============================================
        			# 8. Every C steps reset Q^ = Q, actor^ = actor
        			# ================================================
                    self.soft_update(self.critic_target, self.critic, tau)
                    self.soft_update(self.actor_target, self.actor, tau)
                
                ep_reward += reward
                
                previous_observation =  observation
                total_step = total_step+1
                if done or j == self.config.max_step - 1:
                    writer.add_scalar('Q-max', ep_ave_max_q / float(j), global_step=i)
                    writer.add_scalar('Reward', ep_reward, global_step=i)
                    
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, Average reward: {:.8f}'.format(i, ep_reward, (ep_ave_max_q / float(j)),moving_average_reward))
                    q_max = ep_ave_max_q / float(j)
                    break
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
            moving_average_reward = 0.05 * ep_reward + (1 - 0.05) * moving_average_reward
            writer.add_scalar('Moving average reward', moving_average_reward, global_step=i)
            policy_loss = self.policy_learn(eps)
            writer.add_scalar('Policy Loss', policy_loss, global_step=i)
        all_fpv = np.array(all_fpv)
        all_plot_action = np.array(all_plot_action)
        np.save(f'backtest_result/{self.current_agent.name}_val_record_QPL_{self.config.qpl_level}_{self.config.data_dir}', all_fpv)
        np.save(f'backtest_result/{self.current_agent.name}_plot_action_QPL_{self.config.qpl_level}_{self.config.data_dir}', all_plot_action)
        print('Finish.')
        # torch.save(self.actor.state_dict(), path_join(self.config.ddpg_model_dir, f'{self.current_agent.name}_QPL_{self.config.qpl_level}_{self.config.data_dir}'))
        # torch.save(self.policy.state_dict(), path_join(self.config.pga_model_dir, f'{self.current_agent.name}_QPL_{self.config.qpl_level}_{self.config.data_dir}'))