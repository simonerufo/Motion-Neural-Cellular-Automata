import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
from .utils import create_seed, get_com, shift_target
from .physics import generate_chemotaxis_goal, rotate_target, apply_slash_damage


def train_morphogenesis(model, base_target, epochs=8000, grid_size=64, batch_size=8, device=None):
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    pool = create_seed(size=grid_size, batch_size=256).to(device)
    
    print(f"--- Starting Phase 1: Morphogenesis on {device} ---")
    model.train()
    
    loss_history = [] 
    for i in range(epochs):
        batch_indices = random.sample(range(256), batch_size)
        x = pool[batch_indices].clone().to(device)
        x[:1] = create_seed(size=grid_size, batch_size=1).to(device)
        
        optimizer.zero_grad()
        num_steps = random.randint(64, 96)
        
        for _ in range(num_steps):
            x = model(x)
            
        target_batch = base_target.repeat(batch_size, 1, 1, 1)
        loss = F.mse_loss(x[:, :4, :, :], target_batch)
        loss.backward()
        
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        optimizer.step()
        pool[batch_indices] = x.detach()
        
        if i % 100 == 0:
            print(f"Epoch {i:05d} | Loss: {loss.item():.4f}")
            loss_history.append(loss.item())

    plt.figure(figsize=(6, 4))
    plt.plot(range(0, epochs, 100), loss_history, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Morphogenesis Training Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('loss_morphogenesis.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    return model

def train_chemotaxis(model, base_target, epochs=10000, grid_size=64, batch_size=8, device=None):
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    pool = create_seed(size=grid_size, batch_size=256).to(device)
    
    print(f"--- Starting Phase 2: Chemotaxis on {device} ---")
    model.train()
    loss_history = []
    for i in range(epochs):
        batch_indices = random.sample(range(256), batch_size)
        x = pool[batch_indices].clone().to(device)
        x[:1] = create_seed(size=grid_size, batch_size=1).to(device)

        optimizer.zero_grad()
        loss = 0.0
        
        com_x, com_y = get_com(x, device, grid_size)
        chem_target_x = torch.randint(0, grid_size, (batch_size,), device=device).float()
        chem_target_y = torch.randint(0, grid_size, (batch_size,), device=device).float()
        
        stationary_mask = (torch.rand(batch_size, device=device) < 0.25)
        chem_target_x[stationary_mask] = com_x[stationary_mask]
        chem_target_y[stationary_mask] = com_y[stationary_mask]
        
        num_steps_total = random.randint(64, 96)
        goal_vector, shifts = generate_chemotaxis_goal(com_x, com_y, chem_target_x, chem_target_y, num_steps_total) 
        
        steps_per_segment = num_steps_total // 4
        for segment in range(4):
            for _ in range(steps_per_segment):
                x = model(x, goal=goal_vector, fire_rate=0.5)
                
            current_shift_x, current_shift_y = shifts[segment]
            shifted_targets = []
            for b in range(batch_size):
                align_x = int(round(com_x[b].item())) - (grid_size // 2)
                align_y = int(round(com_y[b].item())) - (grid_size // 2)
                tot_shift_x = align_x + current_shift_x[b].item()
                tot_shift_y = align_y + current_shift_y[b].item()
                shifted_targets.append(shift_target(base_target, tot_shift_x, tot_shift_y))

            shifted_target_batch = torch.cat(shifted_targets, dim=0)
            loss += F.mse_loss(x[:, :4, :, :], shifted_target_batch)
        
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        optimizer.step()
        pool[batch_indices] = x.detach()
        
        if i % 100 == 0:
            print(f"Epoch {i:05d} | Loss: {loss.item():.4f}")
            loss_history.append(loss.item())
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(0, epochs, 100), loss_history, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Chemotaxis Training Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('loss_chemotaxis.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    return model

def train_autonomous_life(model, base_target, epochs=10000, grid_size=64, batch_size=8, device=None):
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    pool = create_seed(size=grid_size, batch_size=256).to(device)
    
    print(f"--- Starting Phase 2: Ecosystem/Mitosis on {device} ---")
    model.train()
    loss_history = []
    for i in range(epochs):
        batch_indices = random.sample(range(256), batch_size)
        x = pool[batch_indices].clone().to(device)
        
        x = apply_slash_damage(x, grid_size)
        x[:1] = create_seed(size=grid_size, batch_size=1).to(device)
        
        optimizer.zero_grad()
        loss = 0.0
        
        com_x, com_y = get_com(x, device, grid_size)
        target_angles = (torch.rand(batch_size, device=device) * 2 * math.pi) - math.pi
        goal_vector = torch.stack([torch.cos(target_angles), torch.sin(target_angles)], dim=1)
        
        num_steps = random.randint(64, 96)
        steps_per_segment = num_steps // 4
        speed = 0.75 
        
        for segment in range(4):
            for _ in range(steps_per_segment):
                x = model(x, goal=goal_vector, fire_rate=0.5)
                
            shifted_targets = []
            for b in range(batch_size):
                angle = target_angles[b]
                rotated_img = rotate_target(base_target[b:b+1] if base_target.shape[0]>1 else base_target, angle - (math.pi / 2.0))
                
                forward_x = torch.cos(angle) * speed * (segment + 1)
                forward_y = torch.sin(angle) * speed * (segment + 1)
                
                align_x = int(round(com_x[b].item() + forward_x.item())) - (grid_size // 2)
                align_y = int(round(com_y[b].item() + forward_y.item())) - (grid_size // 2)
                
                shifted_targets.append(torch.roll(rotated_img, shifts=(align_y, align_x), dims=(2, 3)))
                
            shifted_target_batch = torch.cat(shifted_targets, dim=0)
            loss += F.mse_loss(x[:, :4, :, :], shifted_target_batch)
            
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
        optimizer.step()
        pool[batch_indices] = x.detach()
        
        if i % 100 == 0:
            print(f"Epoch {i:05d} | Loss: {loss.item():.4f}")
            loss_history.append(loss.item())

    plt.figure(figsize=(6, 4))
    plt.plot(range(0, epochs, 100), loss_history, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Ecosystem Training Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('loss_ecosystem.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    return model
