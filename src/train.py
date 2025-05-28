import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
import numpy as np  

import wandb
from src.model import DecisionTransformer
from data.data_loader import generate_episodes
from data.data_preprocess import preprocess_dataset
from data.sequence_dataset import DecisionTransformerDataset
from src.utils import seed_everything

def train_model(
    dataset,
    state_dim,
    action_dim,
    embed_dim=128,
    context_len=20,
    epochs=10,
    batch_size=64,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    seed_everything(42)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        context_length=context_len
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    wandb.init(project="decision-transformer", config={
        "embed_dim": embed_dim,
        "context_len": context_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            rtgs, states, actions, timesteps = [x.to(device) for x in batch]

            pred_actions = model(rtgs, states, actions, timesteps)
            loss = loss_fn(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())  

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

       #save model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

    return model


if __name__ == "__main__":
    print("Generating episodes.")
    raw_episodes = generate_episodes(env_name="HalfCheetah-v5", num_episodes=100)

    print("Preprocessing.")
    context_len = 10
    sequences, state_mean, state_std, action_mean, action_std = preprocess_dataset(raw_episodes, K=context_len)

    print("Preparing dataset.")
    dataset = DecisionTransformerDataset(sequences, context_len)

    print("Starting training.")
    model = train_model(
        dataset=dataset,
        state_dim=17,
        action_dim=6,
        embed_dim=64,
        context_len=context_len,
        epochs=3,
        batch_size=16,
        learning_rate=1e-4
    )

    # save normalization stats
    np.savez("normalization_stats.npz",
             state_mean=state_mean,
             state_std=state_std,
             action_mean=action_mean,
             action_std=action_std)

    print("Training complete. Model and stats saved.")
