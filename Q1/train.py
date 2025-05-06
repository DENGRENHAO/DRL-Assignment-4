import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds):
        super(PolicyNetwork, self).__init__()
        
        self.action_bounds = action_bounds
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        action = action * self.action_bounds
        
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)
        
        self.fc3 = nn.Linear(state_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2

class SAC:
    def __init__(
        self, 
        state_dim, 
        action_dim,
        action_bounds,
        lr=0.001,
        gamma=0.99,
        tau=0.05,
        alpha=0.2,
        batch_size=256,
        buffer_capacity=100000,
        auto_tune_alpha=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.memory = ReplayBuffer(buffer_capacity, self.device)
        
        self.actor = PolicyNetwork(state_dim, action_dim, action_bounds).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.critic_target = QNetwork(state_dim, action_dim).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        if auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
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
                action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()
    
    def train(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actions_current, log_probs = self.actor.sample(states)
        q1_current, q2_current = self.critic(states, actions_current)
        q_current = torch.min(q1_current, q2_current)
        
        actor_loss = (self.alpha * log_probs - q_current).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        
def plot(episode_rewards, window=10):
    episode_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title('SAC Training on Pendulum-v1')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('./models/train.png')
    plt.show()
    

def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bounds = float(env.action_space.high[0])
    
    max_episodes = 500
    max_steps = 200
    train_start = 1000
    
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    agent = SAC(state_dim, action_dim, action_bounds)
    
    episode_rewards = []
    
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            
            state = next_state
            episode_reward += reward
            
            if len(agent.memory) > train_start:
                agent.train()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode}/{max_episodes} | Reward: {episode_reward:.2f}")
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{max_episodes} | Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")
            plot(episode_rewards)

        if episode % 50 == 0:
            agent.save(f"./models/train_ep{episode}.pth")
        
    
    env.close()
    
    return agent

if __name__ == "__main__":
    train()