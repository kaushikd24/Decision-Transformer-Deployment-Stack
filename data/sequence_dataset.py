import torch
from torch.utils.data import Dataset

class DecisionTransformerDataset(Dataset):
    def __init__(self, sequences, K):
        self.sequences = sequences
        self.K = K

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        rtg_seq, state_seq, action_seq, _ = self.sequences[idx]
        
        return (
            torch.tensor(rtg_seq, dtype=torch.float32).reshape(self.K, 1),
            torch.tensor(state_seq, dtype=torch.float32),
            torch.tensor(action_seq, dtype=torch.float32),
            torch.arange(self.K, dtype=torch.long)
        )
