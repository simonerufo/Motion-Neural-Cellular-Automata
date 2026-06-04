import torch
import torch.nn as nn
import torch.nn.functional as F

class CAModel(nn.Module):
    def __init__(self, channel_n=16, hidden_size=128, goal_dim=2, circular_padding=False):
        super(CAModel, self).__init__()
        self.channel_n = channel_n
        self.circular_padding = circular_padding # Switch for toroidal world (donut grid)
        
        # --- 1. PERCEPTION SETUP ---
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], 
                                [ 0.0,  0.0,  0.0], 
                                [ 1.0,  2.0,  1.0]]) / 8.0
        
        weight_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1)
        weight_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(channel_n, 1, 1, 1)
        
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        # --- 2. NCA UPDATE NETWORK ---
        self.fc1 = nn.Conv2d(channel_n * 3, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_size, channel_n, kernel_size=1)
        
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # --- 3. GOAL ENCODER ---
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, channel_n - 4) 
        )

        nn.init.zeros_(self.goal_encoder[-1].weight)
        nn.init.zeros_(self.goal_encoder[-1].bias)
    
    def perceive(self, x):
        # Switch to choose the boundary physics
        if self.circular_padding:
            x_pad = F.pad(x, (1, 1, 1, 1), mode='circular')
            grad_x = F.conv2d(x_pad, self.weight_x, padding=0, groups=self.channel_n)
            grad_y = F.conv2d(x_pad, self.weight_y, padding=0, groups=self.channel_n)
        else:
            grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.channel_n)
            grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.channel_n)
        return torch.cat([x, grad_x, grad_y], dim=1)

    def forward(self, x, goal=None, fire_rate=0.5):
        if goal is not None:
            perturbation = self.goal_encoder(goal) 
            perturbation = F.pad(perturbation, (4, 0), "constant", 0) 
            perturbation = perturbation.unsqueeze(-1).unsqueeze(-1)
            
            alive_mask = F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1
            x = x + (perturbation * alive_mask.float())

        y = self.perceive(x)
        dx = self.fc1(y)
        dx = F.relu(dx)
        dx = self.fc2(dx)
        
        update_mask = (torch.rand(x[:, :1, :, :].shape, device=x.device) <= fire_rate).float()
        x_new = x + dx * update_mask         
        
        alive_mask = F.max_pool2d(x_new[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1
        x_new = x_new * alive_mask.float()
        x_new = torch.clamp(x_new, min=-10.0, max=10.0)
        
        return x_new
