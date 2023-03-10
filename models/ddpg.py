import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
from environment.QF_env import envs
from tools.ddpg.replay_buffer import ReplayBuffer
from api_types import GlobalConfig, AgentProps
from tools.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from typing import Callable, List, cast, OrderedDict, Union, Any
from os.path import join as path_join, exists as path_exists

def hidden_init(layer):
    # Initialize the parameter of hidden layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / math.sqrt(fan_in)
    return (-lim, lim)


# Define actor network
class Actor(nn.Module):
    def __init__(self,product_num, win_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels =  1,
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
        fc3_out = F.softmax(fc3_out,dim=1)
        
        return fc3_out

# Define Critic network
class Critic(nn.Module):
    def __init__(self, product_num, win_size):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels =  1,
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


class DDPG(object):
    
    config: GlobalConfig
    actor_noise: Callable
    summary_path: str = path_join('train_results', 'ddpg')
    current_agent: AgentProps
    price_history: pd.DataFrame
    trading_dates: List[str]
    window_size: int
    use_cuda: bool


    def __init__(self, env, window_size: int, actor_noise: Callable, config: GlobalConfig):
        
        # Load configuration
        self.config = config
        assert self.config is not None, "Can't load config"

        assert type(config.use_agents) == int, "You must specify one agent for training!"
        agent_index = cast(int, config.use_agents)
        self.current_agent = AgentProps(config.agent_list[agent_index])
        product_num = len(self.current_agent.product_list)
        


        self.env = env
        self.actor_noise = actor_noise
        self.summary_path ='results/ddpg/'
        if C_CUDA:
            self.actor = Actor(product_num,win_size).cuda()
            self.actor_target = Actor(product_num,win_size).cuda()
            self.critic = Critic(product_num,win_size).cuda()
            self.critic_target = Critic(product_num,win_size).cuda()
        else:
            self.actor = Actor(product_num,win_size)
            self.actor_target = Actor(product_num,win_size)
            self.critic = Critic(product_num,win_size)
            self.critic_target = Critic(product_num,win_size)
        
        self.actor.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_target.reset_parameters()
        self.actor.reset_parameters()
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.config['actor learning rate'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.config['critic learning rate'])
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def act(self, state):
        if C_CUDA:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).cuda()
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().detach().numpy()+ self.actor_noise()
        return action
    
    def critic_learn(self, state, action, predicted_q_value):
        actual_q = self.critic(state, action)
        if C_CUDA:
            target_Q = torch.tensor(predicted_q_value, dtype=torch.float).cuda()
        else:
            target_Q = torch.tensor(predicted_q_value, dtype=torch.float)
        target_Q=Variable(target_Q,requires_grad=True)
        td_error  = F.mse_loss(actual_q, target_Q)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()
        return predicted_q_value,td_error
    
    def actor_learn(self, state):

        loss = -self.critic(state, self.actor(state)).mean()
        

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss
        

    
    def soft_update(self, net_target, net, tau):
        for target_param, param  in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def train(self):
        num_episode = self.config['episode']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        tau = self.config['tau']
        self.buffer = ReplayBuffer(self.config['buffer size'])
        total_step = 0
        writer = SummaryWriter(logdir=self.summary_path)
        # Main training loop
        for i in range(100):
            previous_observation = self.env.reset()
            # Normalization
            previous_observation = obs_normalizer(previous_observation)
            # Reshape
            previous_observation = previous_observation.transpose(2, 0, 1)
            ep_reward = 0
            ep_ave_max_q = 0
            
            # Keep sampling until done
            for j in range (self.config['max step']):
                # ================================================
        		# 1. Given state st, take action at based on actor
        		# ================================================
                action = self.act(previous_observation)
                # ================================================
        		# 2. Obtain reward rt and reach new state st+1
                # ================================================
                observation, reward, done, _ = self.env.step(action)
                observation = obs_normalizer(observation)
                # Reshape
                observation = observation.transpose(2, 0, 1)
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
                    if C_CUDA:
                        s_batch = torch.tensor(s_batch, dtype=torch.float).cuda()
                        a_batch = torch.tensor(a_batch, dtype=torch.float).cuda()
                        r_batch = torch.tensor(r_batch, dtype=torch.float).cuda()#.view(batch_size,-1)
                        t_batch = torch.tensor(t_batch, dtype=torch.float).cuda()
                        s2_batch = torch.tensor(s2_batch, dtype=torch.float).cuda()
                        target_q = self.critic_target(s2_batch,self.actor_target(s2_batch)).cpu().detach()
                    else:
                        s_batch = torch.tensor(s_batch, dtype=torch.float)
                        a_batch = torch.tensor(a_batch, dtype=torch.float)
                        r_batch = torch.tensor(r_batch, dtype=torch.float)
                        t_batch = torch.tensor(t_batch, dtype=torch.float)
                        s2_batch = torch.tensor(s2_batch, dtype=torch.float)
                        target_q = self.critic_target(s2_batch,self.actor_target(s2_batch)).detach()
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
                if done or j == self.config['max step'] - 1:
                    writer.add_scalar('Q-max', ep_ave_max_q / float(j), global_step=i)
                    writer.add_scalar('Reward', ep_reward, global_step=i)
                    
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j))))
                    break
        print('Finish.')
        torch.save(self.actor.state_dict(), model_add+model_name)