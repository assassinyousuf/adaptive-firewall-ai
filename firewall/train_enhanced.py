"""
Enhanced training script with validation split, early stopping, and better hyperparameters.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from firewall.env import FirewallEnv
import os


def load_dataset(filepath="data/traffic_enhanced.csv", fallback="data/traffic.csv"):
    """
    Load training dataset with fallback.
    
    Args:
        filepath: Path to enhanced CSV file
        fallback: Path to fallback CSV file
        
    Returns:
        Numpy array with traffic data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} samples from {filepath}")
        return df.values
    except FileNotFoundError:
        print(f"[WARNING] {filepath} not found, trying {fallback}")
        try:
            df = pd.read_csv(fallback)
            print(f"[INFO] Loaded {len(df)} samples from {fallback}")
            
            # Pad old format (3 features) to new format (7 features)
            if df.shape[1] == 4:  # Old format: [size, protocol, rate, label]
                print("[INFO] Converting old format to new format...")
                # Add zeros for missing features
                df['entropy'] = 0.0
                df['size_variance'] = 0.0
                df['inter_arrival_time'] = 0.0
                df['flags'] = 0.0
                # Reorder columns
                df = df[['packet_size', 'protocol', 'packet_rate', 'entropy',
                        'size_variance', 'inter_arrival_time', 'flags', 'label']]
            
            return df.values
        except FileNotFoundError:
            print(f"[ERROR] Neither dataset found. Generate with:")
            print("  python data/generate_dataset.py")
            return None


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed
        
    Returns:
        train_data, val_data, test_data
    """
    np.random.seed(seed)
    np.random.shuffle(dataset)
    
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train+n_val]
    test_data = dataset[n_train+n_val:]
    
    print(f"\n[INFO] Dataset split:")
    print(f"  Training:   {len(train_data)} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_data)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:       {len(test_data)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_data, val_data, test_data


def train_enhanced_dqn(train_data, val_data, total_timesteps=100000, save_path="model/firewall_dqn_enhanced"):
    """
    Train enhanced DQN agent with validation and early stopping.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        total_timesteps: Number of training steps
        save_path: Where to save the trained model
    """
    print("\n" + "="*60)
    print("TRAINING ENHANCED DQN AGENT")
    print("="*60)
    
    # Create environments
    train_env = FirewallEnv(train_data)
    val_env = FirewallEnv(val_data)
    
    # Wrap environments with Monitor
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    # Early stopping callback
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )
    
    # Evaluation callback with early stopping
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="./model/",
        log_path="./model/logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
        verbose=1
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./model/checkpoints/",
        name_prefix="firewall_dqn_enhanced"
    )
    
    # Create enhanced DQN model with better hyperparameters
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,  # Slightly lower learning rate
        buffer_size=100000,  # Larger buffer
        learning_starts=1000,
        batch_size=64,  # Larger batch size
        tau=0.005,  # Softer target updates
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,  # More exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,  # Lower final epsilon
        verbose=1,
        tensorboard_log="./model/tensorboard/",
        policy_kwargs=dict(net_arch=[128, 128])  # Larger network
    )
    
    print(f"\n[INFO] Starting training for {total_timesteps} timesteps...")
    print("[INFO] Using enhanced hyperparameters:")
    print("  - Learning rate: 3e-4")
    print("  - Buffer size: 100,000")
    print("  - Batch size: 64")
    print("  - Network: [128, 128]")
    print("  - Early stopping: enabled\n")
    
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


def main():
    """Main training pipeline with enhanced features."""
    
    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)
    os.makedirs("model/checkpoints", exist_ok=True)
    os.makedirs("model/logs", exist_ok=True)
    os.makedirs("model/tensorboard", exist_ok=True)
    
    # Load dataset
    dataset = load_dataset()
    if dataset is None:
        return
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1)
    
    # Save test set for later evaluation
    np.save("model/test_data.npy", test_data)
    print("[INFO] Test set saved to model/test_data.npy")
    
    # Train enhanced model
    model = train_enhanced_dqn(
        train_data, 
        val_data, 
        total_timesteps=100000  # More timesteps for larger dataset
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run evaluation: python -m firewall.evaluate_enhanced")
    print("2. Test live: python -m runtime.policy")
    print("\n")


if __name__ == "__main__":
    main()
