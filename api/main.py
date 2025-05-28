"""
Decision Transformer API for HalfCheetah environment.
This module provides a FastAPI application that serves predictions from a trained Decision Transformer model.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.request_schema import InferenceRequest
import torch
import numpy as np

from src.model import DecisionTransformer
from src.utils import load_normalization_stats

# Configuration
MODEL_PATH = "models/model_epoch_4.pt"
NORM_PATH = "models/normalization_stats.npz"
STATE_DIM = 17  # HalfCheetah-v5 state dimension
ACTION_DIM = 6  # HalfCheetah-v5 action dimension
CONTEXT_LEN = 30
EMBED_DIM = 256
N_LAYERS = 6
N_HEADS = 4
DROPOUT = 0.05

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize FastAPI app
app = FastAPI(
    title="Decision Transformer API",
    description="API for serving predictions from a trained Decision Transformer model on HalfCheetah-v5",
    version="1.0.0"
)

try:
    # Initialize model
    model = DecisionTransformer(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        embed_dim=EMBED_DIM,
        context_length=CONTEXT_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load normalization statistics
    state_mean, state_std, action_mean, action_std = load_normalization_stats(NORM_PATH, device)

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

@app.post("/predict", response_model=dict)
async def predict(request: InferenceRequest):
    """
    Generate action predictions using the Decision Transformer model.
    
    Args:
        request (InferenceRequest): Request containing state, past actions, return-to-go, and timesteps
        
    Returns:
        dict: Predicted action vector
        
    Raises:
        HTTPException: If there's an error during prediction
    """
    try:
        # Convert input tensors
        state = (torch.tensor(request.state).float().to(device) - state_mean) / state_std
        rtg = torch.tensor(request.rtg).float().unsqueeze(-1).to(device)
        actions = torch.tensor(request.past_actions).float().to(device)
        timesteps = torch.tensor(request.timesteps).long().to(device)

        # Reshape for batch processing
        state = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        rtg = rtg.unsqueeze(0)                   # [1, T, 1]
        actions = actions.unsqueeze(0)           # [1, T, act_dim]
        timesteps = timesteps.unsqueeze(0)       # [1, T]

        # Generate prediction
        with torch.no_grad():
            pred_action = model(rtg, state, actions, timesteps)[:, -1]
            pred_action = pred_action * action_std + action_mean

        return {"action": pred_action.squeeze().cpu().tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "loaded"}

