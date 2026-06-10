import torch
import torch.nn.functional as F
import numpy as np
import random
import math

def generate_chemotaxis_goal(com_x, com_y, target_x, target_y, num_steps_total=64, obstacles=None):
    dx_attr = target_x - com_x
    dy_attr = target_y - com_y
    norm_attr = torch.sqrt(dx_attr**2 + dy_attr**2) + 1e-8
    dir_x_attr = dx_attr / norm_attr
    dir_y_attr = dy_attr / norm_attr
    
    noise_x = (torch.rand_like(com_x) - 0.5) * 0.1
    noise_y = (torch.rand_like(com_y) - 0.5) * 0.1
    dir_x_attr += noise_x
    dir_y_attr += noise_y
    
    dir_x_rep = torch.zeros_like(com_x)
    dir_y_rep = torch.zeros_like(com_y)
    
    if obstacles is not None:
        repulsion_strength = 200.0 
        for obs_x, obs_y in obstacles:
            dx_obs = com_x - obs_x
            dy_obs = com_y - obs_y
            dist_sq = dx_obs**2 + dy_obs**2
            dist_sq = torch.clamp(dist_sq, min=25.0) 
            force = repulsion_strength / dist_sq
            
            rad_x = (dx_obs / torch.sqrt(dist_sq)) * force
            rad_y = (dy_obs / torch.sqrt(dist_sq)) * force
            
            tang_x = -rad_y * 1.5
            tang_y = rad_x * 1.5
            
            dir_x_rep += (rad_x + tang_x)
            dir_y_rep += (rad_y + tang_y)

    tot_x = dir_x_attr + dir_x_rep
    tot_y = dir_y_attr + dir_y_rep
    
    norm_tot = torch.sqrt(tot_x**2 + tot_y**2) + 1e-8
    dir_x = tot_x / norm_tot
    dir_y = tot_y / norm_tot
    
    goal_vector = torch.stack([dir_x, dir_y], dim=1) 
    
    total_shift_pixels = num_steps_total / 16.0 
    shifts = []
    for step_fraction in [0.25, 0.5, 0.75, 1.0]:
        shift_x = torch.round(dir_x * total_shift_pixels * step_fraction).long()
        shift_y = torch.round(dir_y * total_shift_pixels * step_fraction).long()
        shifts.append((shift_x, shift_y))
        
    return goal_vector, shifts

def rotate_target(target_tensor, angle_radians):
    batch_size = target_tensor.shape[0]
    cos_a = torch.cos(angle_radians) if torch.is_tensor(angle_radians) else np.cos(angle_radians)
    sin_a = torch.sin(angle_radians) if torch.is_tensor(angle_radians) else np.sin(angle_radians)
    
    rotation_matrix = torch.zeros((batch_size, 2, 3), device=target_tensor.device)
    rotation_matrix[:, 0, 0] = cos_a
    rotation_matrix[:, 0, 1] = -sin_a
    rotation_matrix[:, 1, 0] = sin_a
    rotation_matrix[:, 1, 1] = cos_a
    
    grid = F.affine_grid(rotation_matrix, target_tensor.size(), align_corners=False)
    rotated_tensor = F.grid_sample(target_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return rotated_tensor

def apply_slash_damage(x, grid_size):
    batch_size = x.shape[0]
    for b in range(batch_size):
        if random.random() < 0.5:
            cut_x = random.randint(grid_size//3, 2*grid_size//3)
            cut_width = random.randint(4, 8)
            if random.random() < 0.5:
                x[b, :, :, cut_x:cut_x+cut_width] = 0.0 
            else:
                x[b, :, cut_x:cut_x+cut_width, :] = 0.0 
    return x
