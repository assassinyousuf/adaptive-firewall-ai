"""
Feature extraction module for network packets.
Converts raw packet data into normalized state vectors for the RL agent.
Enhanced with entropy, statistical features, and timing patterns.
"""

import numpy as np
from collections import Counter


def calculate_entropy(data):
    """Calculate Shannon entropy of data."""
    if not data or len(data) == 0:
        return 0.0
    
    counter = Counter(data)
    length = len(data)
    entropy = 0.0
    
    for count in counter.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def extract_features(row):
    """
    Converts traffic row into RL state vector with enhanced features.
    
    Args:
        row: Dictionary or array containing packet information
        
    Returns:
        List of features [packet_size, protocol, packet_rate, entropy, 
                         size_variance, inter_arrival_time, flags]
    """
    try:
        if isinstance(row, dict):
            return [
                float(row.get("packet_size", 0)),
                float(row.get("protocol", 0)),
                float(row.get("packet_rate", 0)),
                float(row.get("entropy", 0)),
                float(row.get("size_variance", 0)),
                float(row.get("inter_arrival_time", 0)),
                float(row.get("flags", 0)),
            ]
        else:  # Assume array-like (backward compatible)
            if len(row) >= 7:
                return [float(x) for x in row[:7]]
            elif len(row) >= 3:
                # Old format - add zeros for new features
                return [
                    float(row[0]),  # packet_size
                    float(row[1]),  # protocol
                    float(row[2]),  # packet_rate
                    0.0,  # entropy (unknown)
                    0.0,  # size_variance (unknown)
                    0.0,  # inter_arrival_time (unknown)
                    0.0,  # flags (unknown)
                ]
            else:
                return [0.0] * 7
    except Exception as e:
        print(f"[WARNING] Feature extraction failed: {e}")
        return [0.0] * 7


def extract_features_from_packet(packet, packet_history=None):
    """
    Extract enhanced features from a live Scapy packet.
    
    Args:
        packet: Scapy packet object
        packet_history: List of recent packets for statistical analysis
        
    Returns:
        List of features [packet_size, protocol, packet_rate, entropy, 
                         size_variance, inter_arrival_time, flags]
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
        
        # Calculate entropy from packet payload
        entropy = 0.0
        if packet.haslayer("Raw"):
            payload = bytes(packet["Raw"].load)
            if len(payload) > 0:
                entropy = calculate_entropy(payload[:100])  # First 100 bytes
        
        # Statistical features from history
        packet_rate = 0
        size_variance = 0
        inter_arrival_time = 0
        
        if packet_history and len(packet_history) > 1:
            # Calculate packet rate
            packet_rate = len(packet_history)
            
            # Calculate size variance
            sizes = [len(p) for p in packet_history[-10:]]
            if len(sizes) > 1:
                size_variance = float(np.var(sizes))
            
            # Calculate inter-arrival time
            if hasattr(packet_history[-1], 'time') and hasattr(packet_history[-2], 'time'):
                inter_arrival_time = float(packet_history[-1].time - packet_history[-2].time)
        
        # TCP flags (if applicable)
        flags = 0
        if packet.haslayer("TCP"):
            tcp_flags = packet["TCP"].flags
            # Convert flags to numeric representation
            flags = float(tcp_flags.value if hasattr(tcp_flags, 'value') else 0)
        
        return [
            float(packet_size),
            float(protocol),
            float(min(packet_rate, 100)),  # Cap at 100
            float(entropy),
            float(min(size_variance, 10000)),  # Cap variance
            float(inter_arrival_time),
            float(flags)
        ]
    except Exception as e:
        print(f"[WARNING] Packet feature extraction failed: {e}")
        return [0.0] * 7


def normalize_features(features, max_values=None):
    """
    Normalize features to [0, 1] range with improved bounds.
    
    Args:
        features: List of feature values
        max_values: Optional list of maximum values for normalization
        
    Returns:
        Normalized feature vector
    """
    if max_values is None:
        # Updated max values for 7 features
        max_values = [
            2000,   # packet_size
            2,      # protocol
            100,    # packet_rate
            8,      # entropy (max for byte is ~8 bits)
            10000,  # size_variance
            1.0,    # inter_arrival_time (seconds)
            255     # flags (max TCP flags value)
        ]
    
    normalized = []
    for feat, max_val in zip(features, max_values):
        if max_val > 0:
            normalized.append(min(feat / max_val, 1.0))
        else:
            normalized.append(0.0)
    
    return normalized


def extract_features_batch(data_frame):
    """
    Extract features from entire DataFrame efficiently.
    
    Args:
        data_frame: pandas DataFrame with traffic data
        
    Returns:
        numpy array of features
    """
    features = []
    for _, row in data_frame.iterrows():
        features.append(extract_features(row))
    
    return np.array(features)
