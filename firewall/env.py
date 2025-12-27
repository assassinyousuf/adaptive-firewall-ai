"""
Reinforcement Learning Environment for Adaptive Firewall.
Implements Gymnasium interface for training DQN agents.

This is the CORE of the research paper - where the AI learns.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from firewall.rewards import calculate_reward


class FirewallEnv(gym.Env):
    """
    Custom Gym Environment for Firewall Traffic Classification.
    
    State Space: [packet_size, protocol, packet_rate]
    Action Space: {0: Allow, 1: Block}
    
    The environment simulates a firewall making decisions on network traffic.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, dataset, render_mode=None):
        """
        Initialize the Firewall Environment.
        
        Args:
            dataset: Numpy array of shape (N, 4) where columns are:
                     [packet_size, protocol, packet_rate, label]
            render_mode: Optional rendering mode
        """
        super(FirewallEnv, self).__init__()
        
        self.dataset = dataset
        self.current_index = 0
        self.render_mode = render_mode
        
        # Observation space: 3 continuous features
        self.observation_space = spaces.Box(
            low=0, 
            high=2000, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Action space: 0 = Allow, 1 = Block
        self.action_space = spaces.Discrete(2)
        
        # Tracking metrics
        self.episode_rewards = []
        self.decisions = []
        
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset to start of dataset
        self.current_index = 0
        self.episode_rewards = []
        self.decisions = []
        
        # Get initial state (first 3 columns)
        observation = self.dataset[self.current_index][:3].astype(np.float32)
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (Allow) or 1 (Block)
            
        Returns:
            observation: Next state
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get ground truth label
        label = int(self.dataset[self.current_index][3])
        
        # Calculate reward
        reward = calculate_reward(action, label)
        self.episode_rewards.append(reward)
        self.decisions.append((action, label))
        
        # Move to next sample
        self.current_index += 1
        
        # Check if episode is done
        terminated = self.current_index >= len(self.dataset)
        truncated = False
        
        # Get next observation
        if not terminated:
            observation = self.dataset[self.current_index][:3].astype(np.float32)
        else:
            observation = np.zeros(3, dtype=np.float32)
        
        # Info dictionary
        info = {
            "total_reward": sum(self.episode_rewards),
            "decisions_made": len(self.decisions),
            "action": action,
            "label": label
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment state (optional)."""
        if self.render_mode == "human":
            print(f"Step {self.current_index}/{len(self.dataset)}")
            if self.decisions:
                last_action, last_label = self.decisions[-1]
                print(f"Last Action: {'BLOCK' if last_action == 1 else 'ALLOW'}")
                print(f"Correct: {last_action == last_label}")
    
    def close(self):
        """Clean up resources."""
        pass


class FirewallEnvV2(gym.Env):
    """
    Enhanced version with dynamic traffic generation.
    Useful for continuous training without dataset limitations.
    """
    
    def __init__(self):
        super(FirewallEnvV2, self).__init__()
        
        self.observation_space = spaces.Box(
            low=0, 
            high=2000, 
            shape=(3,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        
        self.steps_taken = 0
        self.max_steps = 1000
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # Generate random initial state
        observation = self._generate_traffic()
        return observation, {}
    
    def step(self, action):
        # Simulate traffic and label
        observation = self._generate_traffic()
        label = self._simulate_label(observation)
        
        reward = calculate_reward(action, label)
        
        self.steps_taken += 1
        terminated = self.steps_taken >= self.max_steps
        truncated = False
        
        info = {"action": action, "label": label}
        
        return observation, reward, terminated, truncated, info
    
    def _generate_traffic(self):
        """Generate synthetic traffic data."""
        packet_size = np.random.randint(64, 1500)
        protocol = np.random.randint(0, 3)
        packet_rate = np.random.randint(1, 100)
        
        return np.array([packet_size, protocol, packet_rate], dtype=np.float32)
    
    def _simulate_label(self, observation):
        """Simulate whether traffic is malicious (simple heuristic)."""
        packet_size, protocol, packet_rate = observation
        
        # Simple rule: high rate + large packets = likely malicious
        if packet_rate > 70 and packet_size > 1000:
            return 1  # Malicious
        
        return 0  # Benign
