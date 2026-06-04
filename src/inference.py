import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
from .utils import create_seed, get_com
from .physics import generate_chemotaxis_goal

def run_inference_phase1(model, device, grid_size=64):
    x = create_seed(size=grid_size, batch_size=1, center_x=grid_size//2, center_y=grid_size//2).to(device)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Phase 1 - Morphogenesis")
    
    def to_rgb_fase1(state):
        rgba = state[0, :4].permute(1, 2, 0).cpu().detach().numpy()
        rgba = np.clip(rgba, 0.0, 1.0)
        return np.flipud(rgba[..., :3] * rgba[..., 3:4])
        
    img_plot = ax.imshow(to_rgb_fase1(x), origin='lower')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    def update_fase1(frame):
        nonlocal x
        with torch.no_grad():
            for _ in range(4): x = model(x)
        img_plot.set_data(to_rgb_fase1(x))
        return img_plot,
        
    ani = FuncAnimation(fig, update_fase1, frames=500, interval=50, blit=False)
    plt.show()

def run_inference_chemotaxis(model, device, grid_size=96, use_obstacles=True):
    x = create_seed(size=grid_size, batch_size=1, center_x=grid_size//2, center_y=grid_size//2).to(device)
    chem_target = torch.tensor([80.0, 80.0], device=device)
    obstacles_list = [(30.0, 50.0), (60.0, 30.0), (70.0, 70.0)] if use_obstacles else []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("NCA - Chemotaxis with Obstacles" if use_obstacles else "NCA - Chemotaxis")
    
    def to_rgb_fase2(state, target_coords):
        rgba = state[0, :4].permute(1, 2, 0).cpu().detach().numpy()
        rgba = np.clip(rgba, 0.0, 1.0)
        rgb, alpha = rgba[..., :3], rgba[..., 3:4]
        
        tx, ty = target_coords[0].item(), target_coords[1].item()
        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        
        chem_intensity = np.exp(-((X - tx)**2 + (Y - ty)**2) / (2 * 15.0**2))
        obs_intensity = np.zeros((grid_size, grid_size))
        for ox, oy in obstacles_list:
            obs_intensity += np.exp(-((X - ox)**2 + (Y - oy)**2) / (2 * 6.0**2)) 
                
        bg = np.zeros((grid_size, grid_size, 3))
        bg[..., 0] = np.clip(chem_intensity * 0.9 - obs_intensity, 0, 1) 
        bg[..., 1] = np.clip(chem_intensity * 0.3 + obs_intensity * 0.5, 0, 1) 
        bg[..., 2] = np.clip(0.15 + obs_intensity * 0.8, 0, 1)
        
        return np.flipud(rgb * alpha + bg * (1.0 - alpha))

    img_plot = ax.imshow(to_rgb_fase2(x, chem_target), origin='lower')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    def update_fase2(frame):
        nonlocal x, chem_target
        with torch.no_grad():
            com_x, com_y = get_com(x, device, grid_size)
            dx_pure = chem_target[0] - com_x
            dy_pure = chem_target[1] - com_y
            norm_pure = torch.sqrt(dx_pure**2 + dy_pure**2) + 1e-8
            
            if norm_pure.item() < 5.0:
                valid = False
                while not valid:
                    chem_target = torch.randint(20, grid_size-20, (2,), device=device).float()
                    valid = all(((chem_target[0]-ox)**2 + (chem_target[1]-oy)**2) > 225.0 for ox, oy in obstacles_list) if obstacles_list else True
                com_x, com_y = get_com(x, device, grid_size)              
            
            goal_vector, _ = generate_chemotaxis_goal(com_x, com_y, chem_target[0], chem_target[1], 64, obstacles_list if obstacles_list else None)
            for _ in range(4): x = model(x, goal=goal_vector, fire_rate=0.5)
                
        img_plot.set_data(to_rgb_fase2(x, chem_target))
        return img_plot,
        
    ani = FuncAnimation(fig, update_fase2, frames=1000, interval=50, blit=False)
    plt.show()

def run_inference_ecosystem(model, device, grid_size=128):
    x = create_seed(size=grid_size, batch_size=1, center_x=grid_size//2, center_y=grid_size//2).to(device)
    current_angle = random.uniform(-math.pi, math.pi)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("NCA Ecosystem - Random Exploration & Mitosis")
    ax.axis('off')
    
    def to_rgb_eco(state):
        rgba = state[0, :4].permute(1, 2, 0).cpu().detach().numpy()
        rgba = np.clip(rgba, 0.0, 1.0)
        rgb, alpha = rgba[..., :3], rgba[..., 3:4]
        bg = np.ones_like(rgb) * 0.05 
        return np.flipud(rgb * alpha + bg * (1.0 - alpha))
        
    img_plot = ax.imshow(to_rgb_eco(x), origin='lower')
    
    def update_eco(frame):
        nonlocal x, current_angle
        with torch.no_grad():
            current_angle += random.uniform(-0.15, 0.15)
            if current_angle > math.pi: current_angle -= 2*math.pi
            if current_angle < -math.pi: current_angle += 2*math.pi
            goal_vector = torch.tensor([[math.cos(current_angle), math.sin(current_angle)]], device=device)
            
            if frame > 0 and frame % 200 == 0:
                print("Mitosis Event triggered!")
                com_x, com_y = get_com(x, device, grid_size)
                cx, cy = int(com_x[0].item()), int(com_y[0].item())
                cut_width = 5 
                start_x, end_x = max(0, cx - cut_width), min(grid_size, cx + cut_width)
                start_y, end_y = max(0, cy - cut_width), min(grid_size, cy + cut_width)
                x[0, :, :, start_x:end_x] = 0.0  
                x[0, :, start_y:end_y, :] = 0.0  
            
            for _ in range(6): x = model(x, goal=goal_vector, fire_rate=0.5)
                
        img_plot.set_data(to_rgb_eco(x))
        return img_plot,
        
    ani = FuncAnimation(fig, update_eco, frames=2000, interval=40, blit=False)
    plt.show()
