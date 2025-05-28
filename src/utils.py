import random
import numpy as np
import torch

def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def load_normalization_stats(path: str, device: str = "cpu"):
    """Load normalization statistics from a .npz file."""
    data = np.load(path)
    return (
        torch.tensor(data['state_mean'], dtype=torch.float32).to(device),
        torch.tensor(data['state_std'], dtype=torch.float32).to(device),
        torch.tensor(data['action_mean'], dtype=torch.float32).to(device),
        torch.tensor(data['action_std'], dtype=torch.float32).to(device)
    )

