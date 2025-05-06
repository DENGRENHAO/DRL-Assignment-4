import gymnasium as gym
import numpy as np
import torch
from train import PolicyNetwork

class Agent(object):
    def __init__(self):
        self.state_dim = 67
        self.action_dim = 21
        self.device = torch.device("cpu")
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        
        # self.load_model("models/train_3_ep3500.pth")
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