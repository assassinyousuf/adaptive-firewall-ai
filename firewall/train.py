"""
Training script for the Adaptive Firewall RL Agent.

This is where the AI learns to distinguish between benign and malicious traffic.
Run this script to train the model before deployment.

Usage:
    python -m firewall.train
"""

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from firewall.env import FirewallEnv
import os


def load_dataset(filepath="data/traffic.csv"):
    """
    Load training dataset.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Numpy array with traffic data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} samples from {filepath}")
        return df.values
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at {filepath}")
        print("[INFO] Generating synthetic dataset...")
        return generate_synthetic_dataset(1000)


def generate_synthetic_dataset(n_samples=1000):
    """
    Generate synthetic traffic data for training.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Numpy array with synthetic traffic
    """
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # 70% benign, 30% malicious
        is_malicious = np.random.random() < 0.3
        
        if is_malicious:
            # Malicious: larger packets, higher rate
            packet_size = np.random.randint(800, 1500)
            protocol = np.random.randint(0, 3)
            packet_rate = np.random.randint(50, 100)
            label = 1
        else:
            # Benign: normal patterns
            packet_size = np.random.randint(64, 800)
            protocol = np.random.randint(0, 3)
            packet_rate = np.random.randint(1, 50)
            label = 0
        
        data.append([packet_size, protocol, packet_rate, label])
    
    return np.array(data)


def train_dqn(dataset, total_timesteps=50000, save_path="model/firewall_dqn"):
    """
    Train DQN agent on the dataset.
    
    Args:
        dataset: Training data
        total_timesteps: Number of training steps
        save_path: Where to save the trained model
    """
    print("\n" + "="*60)
    print("TRAINING DQN AGENT")
    print("="*60)
    
    # Create environment
    env = FirewallEnv(dataset)
    
    # Create eval environment
    eval_env = FirewallEnv(dataset)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./model/",
        log_path="./model/logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./model/checkpoints/",
        name_prefix="firewall_dqn"
    )
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./model/tensorboard/"
    )
    
    print(f"\n[INFO] Starting training for {total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(save_path)
    print(f"\n[SUCCESS] Model trained and saved to {save_path}")
    
    return model


def train_ppo(dataset, total_timesteps=50000, save_path="model/firewall_ppo"):
    """
    Train PPO agent (alternative to DQN, often more stable).
    
    Args:
        dataset: Training data
        total_timesteps: Number of training steps
        save_path: Where to save the trained model
    """
    print("\n" + "="*60)
    print("TRAINING PPO AGENT (Alternative)")
    print("="*60)
    
    env = FirewallEnv(dataset)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./model/tensorboard/"
    )
    
    print(f"\n[INFO] Starting PPO training for {total_timesteps} timesteps...")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(save_path)
    
    print(f"\n[SUCCESS] PPO model saved to {save_path}")
    
    return model


def main():
    """Main training pipeline."""
    
    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)
    os.makedirs("model/checkpoints", exist_ok=True)
    os.makedirs("model/logs", exist_ok=True)
    os.makedirs("model/tensorboard", exist_ok=True)
    
    # Load or generate dataset
    dataset = load_dataset()
    
    # Train DQN model
    model = train_dqn(dataset, total_timesteps=50000)
    
    # Optionally train PPO
    # model_ppo = train_ppo(dataset, total_timesteps=50000)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run evaluation: python -m firewall.evaluate")
    print("2. Test live: python -m runtime.policy")
    print("\n")


if __name__ == "__main__":
    main()
