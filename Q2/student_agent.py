import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from train import PolicyNetwork

class Agent(object):
    def __init__(self):
        self.state_dim = 5
        self.action_dim = 1
        self.device = torch.device("cpu")
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        
        # self.load_model("models/train_ep100.pth")
        self.load_model("best_weight.pth")
        
        self.policy.eval()
    
    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        actor_state = state_dict['actor']
        self.policy.load_state_dict(actor_state)
    
    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.policy(state)
        
        return action.numpy().flatten()