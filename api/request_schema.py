from pydantic import BaseModel, Field
from typing import List

class InferenceRequest(BaseModel):
    """Request schema for model inference."""
    
    state: List[float] = Field(
        ...,  # ... means required
        min_items=17,
        max_items=17,
        description="Current state vector of the HalfCheetah environment"
    )
    
    past_actions: List[List[float]] = Field(
        ...,
        description="List of previous actions, each action being a 6-dimensional vector"
    )
    
    rtg: List[float] = Field(
        ...,
        description="Return-to-go values for each timestep"
    )
    
    timesteps: List[int] = Field(
        ...,
        description="Timestep indices"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "state": [0.0] * 17,
                "past_actions": [[0.0] * 6],
                "rtg": [1000.0],
                "timesteps": [0]
            }
        }
