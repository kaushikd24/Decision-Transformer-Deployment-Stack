#preprocess the dataset
import numpy as np

def preprocess_dataset(dataset, K=20):
    
    all_states = np.concatenate([ep["states"] for ep in dataset], axis = 0)
    all_actions = np.concatenate([ep["actions"] for ep in dataset], axis = 0)
    
    #computing the mean/std
    state_mean, state_std = all_states.mean(0), all_states.std(0) + 1e-6
    action_mean, action_std = all_actions.mean(0), all_actions.std(0) + 1e-6
    
    #normalise and extract sequences (fodder for the decision transformer)
    sequences = []
    
    for ep in dataset:
        states = (ep["states"] - state_mean)/state_std
        actions = (ep["actions"] - action_mean)/action_std
        rtgs = ep["rtgs"]
        
        T = len(states)
        
        for t in range(T-K):
            rtg_seq = rtgs[t:t+K]
            state_seq = states[t:t+K]
            action_seq = actions[t:t+K]
            
            # input_seq = np.stack([rtg_seq, state_seq, action_seq], axis = 1) #this is not yet final
            label_seq = action_seq #actions our transformer will predict
            
            sequences.append((rtg_seq.reshape(-1, 1), state_seq, action_seq, action_seq))  # action_seq as label            
    
    return sequences, state_mean, state_std, action_mean, action_std