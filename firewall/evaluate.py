"""
Evaluation script for trained models.

Tests the agent's performance on the dataset and generates metrics.

Usage:
    python -m firewall.evaluate
"""

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from firewall.env import FirewallEnv
from firewall.rewards import calculate_stats
import matplotlib.pyplot as plt


def load_model(model_path="model/firewall_dqn"):
    """
    Load trained model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    try:
        model = DQN.load(model_path)
        print(f"[SUCCESS] Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model not found at {model_path}")
        print("[INFO] Please train a model first: python -m firewall.train")
        return None


def evaluate_model(model, dataset, n_episodes=1):
    """
    Evaluate model performance.
    
    Args:
        model: Trained RL model
        dataset: Test dataset
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    env = FirewallEnv(dataset)
    
    all_predictions = []
    all_labels = []
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            
            all_predictions.append(info["action"])
            all_labels.append(info["label"])
        
        total_rewards.append(episode_reward)
        print(f"[INFO] Episode {episode + 1}/{n_episodes} | Reward: {episode_reward:.2f}")
    
    # Calculate statistics
    stats = calculate_stats(all_predictions, all_labels)
    stats["mean_reward"] = np.mean(total_rewards)
    stats["total_samples"] = len(all_predictions)
    
    return stats


def print_evaluation_results(stats):
    """
    Print formatted evaluation results.
    
    Args:
        stats: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTotal Samples Evaluated: {stats['total_samples']}")
    print(f"Mean Episode Reward: {stats['mean_reward']:.2f}\n")
    
    print("Classification Metrics:")
    print(f"  Accuracy:  {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"  Recall:    {stats['recall']:.4f}")
    print(f"  F1 Score:  {stats['f1_score']:.4f}\n")
    
    print("Confusion Matrix:")
    print(f"  True Positives:  {stats['true_positives']}")
    print(f"  False Positives: {stats['false_positives']}")
    print(f"  True Negatives:  {stats['true_negatives']}")
    print(f"  False Negatives: {stats['false_negatives']}")
    print("="*60 + "\n")


def plot_results(stats):
    """
    Generate visualization of results.
    
    Args:
        stats: Evaluation statistics
    """
    try:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [
            stats['accuracy'],
            stats['precision'],
            stats['recall'],
            stats['f1_score']
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model/evaluation_results.png', dpi=150)
        print("[INFO] Results plot saved to model/evaluation_results.png")
        
    except Exception as e:
        print(f"[WARNING] Could not generate plot: {e}")


def main():
    """Main evaluation pipeline."""
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load dataset
    try:
        df = pd.read_csv("data/traffic.csv")
        dataset = df.values
        print(f"[INFO] Loaded {len(dataset)} samples for evaluation")
    except FileNotFoundError:
        print("[ERROR] Dataset not found. Please create data/traffic.csv first")
        return
    
    # Evaluate
    stats = evaluate_model(model, dataset, n_episodes=1)
    
    # Print results
    print_evaluation_results(stats)
    
    # Plot results
    plot_results(stats)
    
    # Save results to file
    results_text = f"""
Adaptive Firewall AI - Evaluation Results
==========================================

Total Samples: {stats['total_samples']}
Mean Reward: {stats['mean_reward']:.2f}

Metrics:
--------
Accuracy:  {stats['accuracy']:.4f}
Precision: {stats['precision']:.4f}
Recall:    {stats['recall']:.4f}
F1 Score:  {stats['f1_score']:.4f}

Confusion Matrix:
-----------------
True Positives:  {stats['true_positives']}
False Positives: {stats['false_positives']}
True Negatives:  {stats['true_negatives']}
False Negatives: {stats['false_negatives']}
"""
    
    with open("model/evaluation_report.txt", "w") as f:
        f.write(results_text)
    
    print("[SUCCESS] Evaluation report saved to model/evaluation_report.txt")


if __name__ == "__main__":
    main()
