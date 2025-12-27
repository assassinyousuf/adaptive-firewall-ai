"""
Enhanced dataset generation with realistic attack patterns.
Creates a larger, more diverse dataset for better training.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_realistic_dataset(n_samples=5000, seed=42):
    """
    Generate realistic network traffic dataset with diverse patterns.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        pandas DataFrame with traffic data
    """
    np.random.seed(seed)
    
    data = []
    
    # Calculate how many of each type
    n_benign = int(n_samples * 0.7)  # 70% benign
    n_malicious = n_samples - n_benign  # 30% malicious
    
    print(f"Generating {n_samples} samples...")
    print(f"  - Benign: {n_benign}")
    print(f"  - Malicious: {n_malicious}")
    
    # Generate benign traffic with various patterns
    benign_types = [
        ("Web Browsing", 0.3),
        ("Email", 0.2),
        ("File Transfer", 0.2),
        ("Video Streaming", 0.15),
        ("DNS Queries", 0.15)
    ]
    
    for traffic_type, proportion in benign_types:
        n_type = int(n_benign * proportion)
        
        for _ in range(n_type):
            if traffic_type == "Web Browsing":
                packet_size = np.random.randint(64, 600)
                protocol = 0  # TCP
                packet_rate = np.random.randint(5, 30)
                entropy = np.random.uniform(3.0, 6.0)  # Moderate entropy
                size_variance = np.random.uniform(100, 1000)
                inter_arrival_time = np.random.uniform(0.01, 0.5)
                flags = np.random.choice([2, 16, 18])  # SYN, ACK, SYN-ACK
                
            elif traffic_type == "Email":
                packet_size = np.random.randint(100, 500)
                protocol = 0  # TCP
                packet_rate = np.random.randint(1, 15)
                entropy = np.random.uniform(4.0, 6.5)
                size_variance = np.random.uniform(50, 500)
                inter_arrival_time = np.random.uniform(0.1, 1.0)
                flags = np.random.choice([2, 16, 24])
                
            elif traffic_type == "File Transfer":
                packet_size = np.random.randint(500, 1400)
                protocol = 0  # TCP
                packet_rate = np.random.randint(20, 45)
                entropy = np.random.uniform(5.0, 7.5)
                size_variance = np.random.uniform(500, 2000)
                inter_arrival_time = np.random.uniform(0.001, 0.05)
                flags = 16  # ACK
                
            elif traffic_type == "Video Streaming":
                packet_size = np.random.randint(800, 1300)
                protocol = np.random.choice([0, 1])  # TCP or UDP
                packet_rate = np.random.randint(30, 50)
                entropy = np.random.uniform(6.0, 7.8)
                size_variance = np.random.uniform(200, 800)
                inter_arrival_time = np.random.uniform(0.01, 0.03)
                flags = 16 if protocol == 0 else 0
                
            else:  # DNS Queries
                packet_size = np.random.randint(50, 150)
                protocol = 1  # UDP
                packet_rate = np.random.randint(1, 10)
                entropy = np.random.uniform(2.0, 5.0)
                size_variance = np.random.uniform(10, 100)
                inter_arrival_time = np.random.uniform(0.1, 2.0)
                flags = 0  # UDP has no flags
            
            data.append([
                packet_size, protocol, packet_rate, entropy,
                size_variance, inter_arrival_time, flags, 0  # label = benign
            ])
    
    # Generate malicious traffic with various attack patterns
    attack_types = [
        ("DDoS Flood", 0.3),
        ("Port Scan", 0.2),
        ("Brute Force", 0.2),
        ("Data Exfiltration", 0.15),
        ("Malware C&C", 0.15)
    ]
    
    for attack_type, proportion in attack_types:
        n_type = int(n_malicious * proportion)
        
        for _ in range(n_type):
            if attack_type == "DDoS Flood":
                packet_size = np.random.randint(1000, 1500)
                protocol = np.random.choice([0, 1, 2])
                packet_rate = np.random.randint(80, 100)
                entropy = np.random.uniform(1.0, 3.0)  # Low entropy (repetitive)
                size_variance = np.random.uniform(10, 100)  # Low variance
                inter_arrival_time = np.random.uniform(0.0001, 0.001)  # Very fast
                flags = np.random.choice([2, 4])  # SYN or RST flood
                
            elif attack_type == "Port Scan":
                packet_size = np.random.randint(40, 100)
                protocol = 0  # TCP
                packet_rate = np.random.randint(60, 90)
                entropy = np.random.uniform(0.5, 2.0)  # Very low
                size_variance = np.random.uniform(5, 50)
                inter_arrival_time = np.random.uniform(0.001, 0.01)
                flags = 2  # SYN scan
                
            elif attack_type == "Brute Force":
                packet_size = np.random.randint(100, 400)
                protocol = 0  # TCP
                packet_rate = np.random.randint(50, 80)
                entropy = np.random.uniform(2.0, 4.0)
                size_variance = np.random.uniform(50, 200)
                inter_arrival_time = np.random.uniform(0.01, 0.1)
                flags = np.random.choice([2, 16, 18])
                
            elif attack_type == "Data Exfiltration":
                packet_size = np.random.randint(1200, 1500)
                protocol = 0  # TCP
                packet_rate = np.random.randint(60, 85)
                entropy = np.random.uniform(6.0, 7.9)  # High entropy (encrypted)
                size_variance = np.random.uniform(100, 500)
                inter_arrival_time = np.random.uniform(0.001, 0.01)
                flags = 16  # ACK
                
            else:  # Malware C&C
                packet_size = np.random.randint(200, 800)
                protocol = np.random.choice([0, 1])
                packet_rate = np.random.randint(40, 70)
                entropy = np.random.uniform(5.0, 7.5)
                size_variance = np.random.uniform(100, 1000)
                inter_arrival_time = np.random.uniform(0.1, 1.0)
                flags = 16 if protocol == 0 else 0
            
            data.append([
                packet_size, protocol, packet_rate, entropy,
                size_variance, inter_arrival_time, flags, 1  # label = malicious
            ])
    
    # Shuffle the data
    np.random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'packet_size', 'protocol', 'packet_rate', 'entropy',
        'size_variance', 'inter_arrival_time', 'flags', 'label'
    ])
    
    print(f"\n✅ Dataset generated: {len(df)} samples")
    print(f"   Benign: {len(df[df['label']==0])}")
    print(f"   Malicious: {len(df[df['label']==1])}")
    
    return df


def save_dataset(df, filepath="data/traffic_enhanced.csv"):
    """Save dataset to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset saved to {filepath}")


if __name__ == "__main__":
    # Generate enhanced dataset
    df = generate_realistic_dataset(n_samples=5000)
    
    # Save to file
    save_dataset(df, "data/traffic_enhanced.csv")
    
    # Show statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(df.describe())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
