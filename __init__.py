"""
Adaptive Firewall AI - Main Package
====================================

An AI-powered adaptive firewall using Reinforcement Learning.

Modules:
    firewall: RL training components (env, features, rewards, train, evaluate)
    runtime: Live deployment components (sniff, policy, firewall_controller)

Usage:
    # Stage 1: Train the model
    python -m firewall.train
    
    # Stage 2: Observe traffic
    python -m runtime.sniff
    
    # Stage 3: Deploy AI firewall
    python -m runtime.policy

For detailed documentation, see README.md
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "AI-Powered Adaptive Firewall using Reinforcement Learning"

# Import key components
from firewall.env import FirewallEnv
from firewall.features import extract_features, extract_features_from_packet
from firewall.rewards import calculate_reward
from runtime.policy import AdaptiveFirewall
from runtime.sniff import PacketObserver

__all__ = [
    "FirewallEnv",
    "extract_features",
    "extract_features_from_packet",
    "calculate_reward",
    "AdaptiveFirewall",
    "PacketObserver"
]
