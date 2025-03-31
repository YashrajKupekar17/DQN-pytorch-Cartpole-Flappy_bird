# Deep Q-Network (DQN) Implementation

This repository contains a PyTorch implementation of various Deep Q-Network (DQN) algorithms applied to the CartPole and FlappyBird environments. The implementation includes vanilla DQN, Double DQN, and Dueling DQN architectures from scratch.

#### FlappyBird-v0 
![FlappyBird Training](https://github.com/YashrajKupekar17/DQN-pytorch-Cartpole-Flappy_bird/blob/main/assets%20/flappybird1-gif.gif)
#### CartPole-v1 
![CartPole Training](https://github.com/YashrajKupekar17/DQN-pytorch-Cartpole-Flappy_bird/blob/main/assets%20/cartpole1-gif-converter.gif)


## Overview

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. Deep Q-Networks combine deep learning with Q-learning to create agents capable of learning complex behaviors directly from high-dimensional sensory inputs.

This project implements and compares three DQN variants:
- **Vanilla DQN**: The original architecture proposed by DeepMind
- **Double DQN**: An improvement that reduces overestimation bias
- **Dueling DQN**: An architecture that separates state value and advantage streams

## Environments

The implementation has been tested on two environments:
1. **CartPole-v1**: A classic control problem where the agent must balance a pole on a cart
2. **FlappyBird-v0**: A game where the agent controls a bird to navigate through pipes

## Project Structure

```
.
├── agent.py                # Main agent class containing training/testing logic
├── dqn.py                  # Implementation of DQN, Double DQN, and Dueling DQN
├── experience_replay.py    # Experience replay memory implementation
├── hyperparameters.yml     # Configuration for different environments and algorithms
└── runs/                   # Directory for storing experiment results
```

## Key Features

- Modular implementation that allows easy switching between DQN variants
- Experience replay memory for improved sample efficiency
- Epsilon-greedy exploration strategy with decay
- Hyperparameter configuration through YAML files
- Metrics tracking and visualization
- Video recording of agent performance

## Code Highlights

### DQN Architecture

The `DQN` class in `dqn.py` implements both standard and dueling network architectures:

```python
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)
            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)
```

### Double DQN Implementation

The implementation of Double DQN in the `optimize` method of the `Agent` class:

```python
if self.enable_double_dqn:
    # First get the actions from policy network
    best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
    
    # Then get the Q values from target network
    target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
else:
    # Standard DQN
    target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
```

### Experience Replay

The `ReplayMemory` class efficiently stores and samples transitions:

```python
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Gymnasium
- OpenCV (for video recording)
- NumPy, Matplotlib

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YashrajKupekar17/DQN-pytorch-Cartpole-Flappy_bird
cd DQN-pytorch-Cartpole-Flappy_bird
```

2. Install dependencies:
```bash
pip install torch gymnasium flappy-bird-gymnasium opencv-python matplotlib numpy
```

### Training

To train an agent on CartPole:
```bash
python agent.py cartpole1 --train
```

To train an agent on FlappyBird:
```bash
python agent.py flappybird1 --train
```

### Testing and Recording Videos

To test a trained agent and record a video:
```bash
python agent.py flappybird1 --record
```

## Results

The implementation achieves:
- **CartPole-v1**: Solves the environment (average reward of 475+ over 100 consecutive episodes)
- **FlappyBird-v0**: Agent learns to navigate through pipes and achieve high scores

## Enviroments 
-  **CartPole-v1**: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
- **FlappyBird-v0**: https://github.com/Talendar/flappy-bird-gym
## Resource limitations 

Due to limited computational resources, I was unable to train FlappyBird for an extended period. I got the model from (https://github.com/johnnycode8/dqn_pytorch)


## Future Improvements

- Implement Prioritized Experience Replay
- Add Noisy Networks for better exploration
- Integrate Rainbow DQN combining multiple improvements
- Implement hyperparameter optimization




