import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random
from .utils import create_seed, get_com
from .physics import generate_chemotaxis_goal

def run_inference_phase1(model, device, grid_size=64, save_path=None):
    x = create_seed(size=grid_size, batch_size=1, center_x=grid_size//2, center_y=grid_size//2).to(device)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Phase 1 - Morphogenesis")
    
    def to_rgb_phase1(state):
        rgba = state[0, :4].permute(1, 2, 0).cpu().detach().numpy()
        rgba = np.clip(rgba, 0.0, 1.0)
        return np.flipud(rgba[..., :3] * rgba[..., 3:4])
        
    img_plot = ax.imshow(to_rgb_phase1(x), origin='lower')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    def update_phase1(frame):
        nonlocal x
        with torch.no_grad():
            for _ in range(4): x = model(x)
        img_plot.set_data(to_rgb_phase1(x))
        
        frames_to_save = [0, 10, 25, 50,100] # 
        if frame in frames_to_save:
            plt.imsave(f'morpho_step_{frame}.png', img_data)
        
        return img_plot,
    
    tot_frames = 100 if save_path else 500
    ani = FuncAnimation(fig, update_phase1, frames=tot_frames, interval=50, blit=False)
    
    if save_path:
        print(f"Generating GIF (Phase 1): {save_path}...")
        ani.save(save_path, writer='pillow', fps=20)
        print("Save completed!")
    else:
        plt.show()
    plt.close(fig)

def run_inference_chemotaxis(model, device, grid_size=96, use_obstacles=True, save_path=None):
    x = create_seed(size=grid_size, batch_size=1, center_x=grid_size//2, center_y=grid_size//2).to(device)
    margin = 15
    chem_target = torch.tensor([
        random.uniform(margin, grid_size - margin),
        random.uniform(margin, grid_size - margin)
    ], device=device).float()
    obstacles_list = []
    if use_obstacles:
        num_obstacles = 3
        for _ in range(num_obstacles):
            valid = False
            while not valid:
                ox = random.uniform(margin, grid_size - margin)
                oy = random.uniform(margin, grid_size - margin)
                
                dist_spawn = math.hypot(ox - grid_size//2, oy - grid_size//2)
                dist_food = math.hypot(ox - chem_target[0].item(), oy - chem_target[1].item())

                
                if dist_spawn > 15.0 and dist_food > 15.0:
                    if all(math.hypot(ox - ex, oy - ey) > 12.0 for ex, ey in obstacles_list):
                        obstacles_list.append((ox, oy))
                        valid = True

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("NCA - Chemotaxis with Obstacles" if use_obstacles else "NCA - Chemotaxis")
    
    def to_rgb_phase2(state, target_coords):
        rgba = state[0, :4].permute(1, 2, 0).cpu().detach().numpy()
        rgba = np.clip(rgba, 0.0, 1.0)
        rgb, alpha = rgba[..., :3], rgba[..., 3:4]
        
        tx, ty = target_coords[0].item(), target_coords[1].item()
        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        
        chem_intensity = np.exp(-((X - tx)**2 + (Y - ty)**2) / (2 * 15.0**2))
        obs_intensity = np.zeros((grid_size, grid_size))
        for ox, oy in obstacles_list:
            obs_intensity += np.exp(-((X - ox)**2 + (Y - oy)**2) / (2 * 6.0**2)) 
        /        
        bg = np.zeros((grid_size, grid_size, 3))
        bg[..., 0] = np.clip(chem_intensity * 0.9 - obs_intensity, 0, 1) 
        bg[..., 1] = np.clip(chem_intensity * 0.3 + obs_intensity * 0.5, 0, 1) 
        bg[..., 2] = np.clip(0.15 + obs_intensity * 0.8, 0, 1)
        
        return np.flipud(rgb * alpha + bg * (1.0 - alpha))

    img_plot = ax.imshow(to_rgb_phase2(x, chem_target), origin='lower')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    def update_phase2(frame):
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
                
        img_plot.set_data(to_rgb_phase2(x, chem_target))

        return img_plot,
        
    tot_frames = 500 if save_path else 1000
    ani = FuncAnimation(fig, update_phase2, frames=tot_frames, interval=50, blit=False)
    
    if save_path:
        print(f"Generating GIF (Chemotaxis): {save_path}...")
        ani.save(save_path, writer='pillow', fps=20)
        print("Save completed!")
    else:
        plt.show()
    plt.close(fig)


def run_inference_ecosystem(model, device, grid_size=128, save_path=None):
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
                
        img_data = to_rgb_eco(x)
        img_plot.set_data(img_data)
 
        frames_to_save = [199, 200, 205, 220]
        if frame in frames_to_save:
            plt.imsave(f'mitosis_step_{frame}.png', img_data)

        return img_plot,
        
    tot_frames = 600 if save_path else 2000
    ani = FuncAnimation(fig, update_eco, frames=tot_frames, interval=40, blit=False)
    
    if save_path:
        print(f"Generating GIF (Ecosystem): {save_path}...")
        ani.save(save_path, writer='pillow', fps=25)
        print("Save completed!")
    else:
        plt.show()
    plt.close(fig)
