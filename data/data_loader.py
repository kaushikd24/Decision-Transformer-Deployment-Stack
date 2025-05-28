import gymnasium as gym
import numpy as np

def generate_episodes (env_name = "HalfCheetah-v5",
                       num_episodes = 10, max_steps = 1000):
    
    env = gym.make(env_name)
    dataset = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode = {"states": [], "actions": [],
                   "rewards": [], "rtgs": []}
        
        total_return = 0
        states, actions, rewards = [], [], []
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            obs = next_obs
            total_return += reward
            
            if done:
                break
        
        rtgs = np.array(np.flip(np.cumsum(np.flip(rewards))))
        
        episode["states"] = np.array(states)
        episode["actions"] = np.array(actions)
        episode["rewards"] =np.array(rewards)
        episode["rtgs"] = rtgs
        
        dataset.append(episode)
    
    return dataset

if __name__ == "__main__":
    
    data = generate_episodes()
    print(f"Collected {len(data)} episodes.")
    
    print("Shape of first episode states:", data[0]["states"].shape)
    
    print("Shape of first episode actions:", data[0]["actions"].shape)
    
    print("Shape of first episode RTGs:", data[0]["rtgs"].shape)
    