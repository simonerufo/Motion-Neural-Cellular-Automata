import os
import torch
import numpy as np
from PIL import Image

def create_seed(size=64, channel_n=16, batch_size=1, center_x=None, center_y=None):
    seed = torch.zeros((batch_size, channel_n, size, size), dtype=torch.float32)
    if center_x is None: center_x = size // 2
    if center_y is None: center_y = size // 2
    seed[:, 3:, center_y, center_x] = 1.0
    return seed

def get_com(x, device, grid_size):
    alpha = x[:, 3:4, :, :] 
    y_grid, x_grid = torch.meshgrid(torch.arange(grid_size, device=device), 
                                    torch.arange(grid_size, device=device), 
                                    indexing='ij')
    mass = alpha.sum(dim=(2, 3)) + 1e-8 
    com_x = (alpha * x_grid).sum(dim=(2, 3)) / mass 
    com_y = (alpha * y_grid).sum(dim=(2, 3)) / mass 
    return com_x.squeeze(1), com_y.squeeze(1)

def shift_target(target_tensor, shift_x, shift_y):
    return torch.roll(target_tensor, shifts=(shift_y, shift_x), dims=(2, 3))

def load_target_image(filepath, size=64, device='cpu'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    img = Image.open(filepath).convert('RGBA')
    img = img.resize((size, size), Image.BILINEAR)
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)
    img_tensor[:, :3, :, :] *= img_tensor[:, 3:4, :, :]
    return img_tensor.to(device)
