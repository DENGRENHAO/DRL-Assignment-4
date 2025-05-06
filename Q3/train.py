import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import os
import sys
from collections import deque
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.buffer_capacity = capacity
        self.device = device
        self.buffer_counter = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        index = self.buffer_counter % self.buffer_capacity
        
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = 1.0 if done else 0.0
        
        self.buffer_counter += 1
    
    def sample(self, batch_size):
        buffer_range = min(self.buffer_counter, self.buffer_capacity)
        batch_index = np.random.choice(buffer_range, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[batch_index]).to(self.device)
        actions = torch.FloatTensor(self.actions[batch_index]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[batch_index].reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(self.next_states[batch_index]).to(self.device)
        dones = torch.FloatTensor(self.dones[batch_index].reshape(-1, 1)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return min(self.buffer_counter, self.buffer_capacity)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 256)
        self.value = nn.Linear(256, 1)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(256)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        value = self.value(x)
        return value

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds=1.0):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.action_bounds = action_bounds
        
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)
        self.layer_norm3 = nn.LayerNorm(256)
        
    def forward(self, state):
        x = F.relu(self.layer_norm1(self.fc1(state)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = F.relu(self.layer_norm3(self.fc3(x)))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -10, 2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state, reparameterize=True):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        if reparameterize:
            x_t = normal.rsample()
        else:
            x_t = normal.sample()
            
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        action = action * self.action_bounds
        
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.q1 = nn.Linear(256, 1)
        
        self.fc4 = nn.Linear(state_dim + action_dim, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 256)
        self.q2 = nn.Linear(256, 1)
        
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        
        self.ln4 = nn.LayerNorm(512)
        self.ln5 = nn.LayerNorm(512)
        self.ln6 = nn.LayerNorm(256)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.ln1(self.fc1(sa)))
        q1 = F.relu(self.ln2(self.fc2(q1)))
        q1 = F.relu(self.ln3(self.fc3(q1)))
        q1 = self.q1(q1)
        
        q2 = F.relu(self.ln4(self.fc4(sa)))
        q2 = F.relu(self.ln5(self.fc5(q2)))
        q2 = F.relu(self.ln6(self.fc6(q2)))
        q2 = self.q2(q2)
        
        return q1, q2

class ImprovedSAC:
    def __init__(
        self, 
        state_dim, 
        action_dim,
        action_bounds=1.0,
        lr=0.0003,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=256,
        buffer_capacity=1000000,
        reward_scale=1.0,
        auto_tune_alpha=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.memory = ReplayBuffer(buffer_capacity, state_dim, action_dim, self.device)
        
        self.actor = PolicyNetwork(state_dim, action_dim, action_bounds).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.value = ValueNetwork(state_dim).to(self.device)
        self.value_target = ValueNetwork(state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(param.data)
        
        if auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = torch.exp(self.log_alpha)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.actor.action_bounds
            return action.cpu().data.numpy().flatten()
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state, reparameterize=False)
            return action.cpu().data.numpy().flatten()

    def update_value_network(self, states):
        with torch.no_grad():
            actions, log_probs = self.actor.sample(states, reparameterize=False)
            q1, q2 = self.critic(states, actions)
            q = torch.min(q1, q2)
            target = q - self.alpha * log_probs
        
        value = self.value(states)
        value_loss = F.mse_loss(value, target)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_optimizer.step()
        
        return value_loss.item()

    def update_actor_and_alpha(self, states):
        actions, log_probs = self.actor.sample(states, reparameterize=True)
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        else:
            alpha_loss = 0
            
        return actor_loss.item(), alpha_loss

    def update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_value = self.value_target(next_states)
            target_q = self.reward_scale * rewards + (1.0 - dones) * self.gamma * next_value
        
        current_q1, current_q2 = self.critic(states, actions)
        
        q1_loss = F.smooth_l1_loss(current_q1, target_q)
        q2_loss = F.smooth_l1_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_target_networks(self):
        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, update_policy=True):
        if len(self.memory) < self.batch_size:
            return None, None, None, None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        critic_loss = self.update_critic(states, actions, rewards, next_states, dones)
        
        value_loss = self.update_value_network(states)
        
        if update_policy:
            actor_loss, alpha_loss = self.update_actor_and_alpha(states)
        else:
            actor_loss, alpha_loss = 0, 0
        
        self.update_target_networks()
            
        return critic_loss, value_loss, actor_loss, alpha_loss
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'value': self.value.state_dict(),
            'value_target': self.value_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'alpha': self.alpha
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.value.load_state_dict(checkpoint['value'])
        self.value_target.load_state_dict(checkpoint['value_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.alpha = checkpoint['alpha']

def plot_training_progress(episode_rewards, avg_rewards, critic_losses, actor_losses, value_losses, alpha_values):
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    axs[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Rewards')
    axs[0, 0].plot(avg_rewards, label='Moving Average (100)')
    axs[0, 0].set_title('Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    if critic_losses:
        axs[0, 1].plot(critic_losses)
        axs[0, 1].set_title('Critic Loss')
        axs[0, 1].set_xlabel('Training Step')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
    
    if value_losses:
        axs[1, 0].plot(value_losses)
        axs[1, 0].set_title('Value Loss')
        axs[1, 0].set_xlabel('Training Step')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True)
    
    if actor_losses:
        axs[1, 1].plot(actor_losses)
        axs[1, 1].set_title('Actor Loss')
        axs[1, 1].set_xlabel('Training Step')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].grid(True)
    
    if alpha_values:
        axs[2, 0].plot(alpha_values)
        axs[2, 0].set_title('Alpha Values')
        axs[2, 0].set_xlabel('Training Step')
        axs[2, 0].set_ylabel('Alpha')
        axs[2, 0].grid(True)
    
    axs[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./models/train_3.png')
    plt.close()

def train():
    env = make_dmc_env("humanoid-walk", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = float(env.action_space.high[0])
    
    print(f"Observation space: {env.observation_space}, Dim: {state_dim}")
    print(f"Action space: {env.action_space}, Dim: {action_dim}")
    
    max_episodes = 10000
    train_start = 5000
    save_interval = 100
    policy_update_interval = 1
    reward_scale = 5.0
    
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    agent = ImprovedSAC(
        state_dim, 
        action_dim,
        action_bounds=action_bounds,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_capacity=1000000,
        reward_scale=reward_scale,
        auto_tune_alpha=True
    )
    
    episode_rewards = []
    avg_rewards = []
    critic_losses = []
    value_losses = []
    actor_losses = []
    alpha_values = []
    
    total_steps = 0
    update_count = 0
    
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        
        episode_critic_losses = []
        episode_value_losses = []
        episode_actor_losses = []
        episode_alpha_values = []

        while not done:        
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if len(agent.memory) > train_start:
                update_policy = update_count % policy_update_interval == 0
                losses = agent.train(update_policy=update_policy)
                update_count += 1
                
                if losses[0] is not None:
                    critic_loss, value_loss, actor_loss, alpha_loss = losses
                    episode_critic_losses.append(critic_loss)
                    episode_value_losses.append(value_loss)
                    
                    if update_policy:
                        episode_actor_losses.append(actor_loss)
                        episode_alpha_values.append(agent.alpha.item())
        
        episode_rewards.append(episode_reward)
        if episode_critic_losses:
            critic_losses.extend(episode_critic_losses)
            value_losses.extend(episode_value_losses)
            actor_losses.extend(episode_actor_losses)
            alpha_values.extend(episode_alpha_values)
        
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        print(f"Episode {episode}/{max_episodes} | Reward: {episode_reward:.2f} | Avg100: {avg_rewards[-1]:.2f} | Steps: {episode_steps}")
                
        if episode % 10 == 0:
            plot_training_progress(
                episode_rewards, 
                avg_rewards, 
                critic_losses, 
                actor_losses, 
                value_losses,
                alpha_values
            )
            
        if episode % save_interval == 0:
            agent.save(f"./models/train_3_ep{episode}.pth")
    
    env.close()
    return agent

if __name__ == "__main__":
    train()