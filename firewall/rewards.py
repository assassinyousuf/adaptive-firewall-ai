"""
Reward calculation module for the RL agent.
Defines the reward function that teaches the agent correct behavior.
"""


def calculate_reward(action, label):
    """
    Calculate reward based on action correctness.
    
    This is the core intelligence function that defines what "good" behavior means.
    
    Args:
        action: Agent's decision (0 = allow, 1 = block)
        label: Ground truth (0 = benign, 1 = malicious)
        
    Returns:
        Reward value (positive for correct, negative for incorrect)
    """
    # Correct decision
    if action == label:
        return +5.0
    
    # Incorrect decision
    # Blocking benign traffic (False Positive) - moderate penalty
    if action == 1 and label == 0:
        return -3.0
    
    # Allowing malicious traffic (False Negative) - severe penalty
    if action == 0 and label == 1:
        return -10.0
    
    return -5.0


def calculate_advanced_reward(action, label, features):
    """
    Advanced reward function with feature-based penalties.
    
    Args:
        action: Agent's decision (0 = allow, 1 = block)
        label: Ground truth (0 = benign, 1 = malicious)
        features: Feature vector [packet_size, protocol, packet_rate]
        
    Returns:
        Calculated reward value
    """
    base_reward = calculate_reward(action, label)
    
    # Add small penalties for high-rate blocking (to avoid over-blocking)
    if action == 1 and features[2] > 50:  # High packet rate
        base_reward -= 1.0
    
    return base_reward


def calculate_stats(predictions, labels):
    """
    Calculate performance statistics.
    
    Args:
        predictions: List of agent actions
        labels: List of ground truth labels
        
    Returns:
        Dictionary with accuracy, precision, recall, F1 score
    """
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    
    total = tp + fp + tn + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }
