import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gymnasium as gym
import numpy as np

from src.model import DecisionTransformer


def pad_tensor(seq, context_len, dim, dtype=torch.float32, device="cpu"):
    out = torch.zeros((context_len, dim), dtype=dtype, device=device)
    if len(seq) > 0:
        out[-len(seq):] = torch.stack(seq)
    return out


def evaluate_model(
    env_name,
    model,
    context_len,
    state_mean,
    state_std,
    action_mean,
    action_std,
    target_return=1200,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    env = gym.make(env_name)
    state, _ = env.reset()
    done = False

    states = []
    actions = []
    rtgs = [target_return]
    timesteps = [0]
    total_reward = 0

    model.eval()

    while not done:
        # Normalize and convert current state
        norm_state = torch.tensor((state - state_mean) / state_std, dtype=torch.float32, device=device)
        states.append(norm_state)

        if len(states) > context_len:
            states = states[-context_len:]
            actions = actions[-context_len:]
            rtgs = rtgs[-context_len:]
            timesteps = timesteps[-context_len:]

        state_tensor = pad_tensor(states, context_len, norm_state.shape[0], device=device).unsqueeze(0)

        if len(actions) == 0:
            action_tensor = torch.zeros((1, context_len, env.action_space.shape[0]), dtype=torch.float32, device=device)
        else:
            action_tensor = pad_tensor(actions, context_len, env.action_space.shape[0], device=device).unsqueeze(0)

        rtg_tensor = pad_tensor([torch.tensor([r], dtype=torch.float32) for r in rtgs], context_len, 1, device=device).unsqueeze(0)
        timestep_tensor = pad_tensor([torch.tensor([t], dtype=torch.long) for t in timesteps], context_len, 1, dtype=torch.long, device=device).squeeze(-1).unsqueeze(0)

        with torch.no_grad():
            pred_action = model(rtg_tensor, state_tensor, action_tensor, timestep_tensor)
        action = pred_action[0, -1].cpu()
        actions.append(action)

        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        total_reward += reward
        done = terminated or truncated

        state = next_state
        rtgs.append(rtgs[-1] - reward)
        timesteps.append(timesteps[-1] + 1)

    env.close()
    print(f"Total Return: {total_reward:.2f}")
    return total_reward


if __name__ == "__main__":
    print("Loading model and stats.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load normalization stats
    stats = np.load("normalization_stats.npz")
    state_mean = stats["state_mean"]
    state_std = stats["state_std"]
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    # Load trained model
    context_len = 10
    model = DecisionTransformer(
        state_dim=17,
        action_dim=6,
        embed_dim=64,
        context_length=context_len
    )
    model.load_state_dict(torch.load("model_epoch_3.pt", map_location=device))
    model = model.to(device)

    print("Evaluating.")
    evaluate_model(
        env_name="HalfCheetah-v5",
        model=model,
        context_len=context_len,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        target_return=1200,
        device=device
    )
