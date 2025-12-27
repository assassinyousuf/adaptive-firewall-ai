"""
Feature extraction module for network packets.
Converts raw packet data into normalized state vectors for the RL agent.
"""

import numpy as np


def extract_features(row):
    """
    Converts traffic row into RL state vector.
    
    Args:
        row: Dictionary or array containing packet information
        
    Returns:
        List of normalized features [packet_size, protocol, packet_rate]
    """
    try:
        if isinstance(row, dict):
            return [
                float(row.get("packet_size", 0)),
                float(row.get("protocol", 0)),
                float(row.get("packet_rate", 0)),
            ]
        else:  # Assume array-like
            return [
                float(row[0]),  # packet_size
                float(row[1]),  # protocol (0=TCP, 1=UDP, 2=ICMP)
                float(row[2]),  # packet_rate
            ]
    except Exception as e:
        print(f"[WARNING] Feature extraction failed: {e}")
        return [0.0, 0.0, 0.0]


def extract_features_from_packet(packet):
    """
    Extract features from a live Scapy packet.
    
    Args:
        packet: Scapy packet object
        
    Returns:
        List of features [packet_size, protocol, packet_rate]
    """
    try:
        # Packet size
        packet_size = len(packet)
        
        # Protocol identification
        protocol = 0
        if packet.haslayer("TCP"):
            protocol = 0
        elif packet.haslayer("UDP"):
            protocol = 1
        elif packet.haslayer("ICMP"):
            protocol = 2
            
        # Placeholder for packet rate (requires time-series tracking)
        packet_rate = 0
        
        return [float(packet_size), float(protocol), float(packet_rate)]
    except Exception as e:
        print(f"[WARNING] Packet feature extraction failed: {e}")
        return [0.0, 0.0, 0.0]


def normalize_features(features, max_values=None):
    """
    Normalize features to [0, 1] range.
    
    Args:
        features: List of feature values
        max_values: Optional list of maximum values for normalization
        
    Returns:
        Normalized feature vector
    """
    if max_values is None:
        max_values = [2000, 2, 100]  # Default max values
    
    normalized = []
    for i, (feat, max_val) in enumerate(zip(features, max_values)):
        normalized.append(min(feat / max_val, 1.0))
    
    return normalized
