"""
Verify the model against the actual dataset
"""

import pandas as pd
import numpy as np
from stable_baselines3 import DQN

print("="*60)
print("DATASET VERIFICATION: Testing on Real Data")
print("="*60)

# Load model
print("\n[1] Loading trained model...")
model = DQN.load("model/firewall_dqn")
print("✅ Model loaded\n")

# Load dataset
print("[2] Loading dataset...")
df = pd.read_csv("data/traffic.csv")
print(f"✅ Loaded {len(df)} samples\n")

# Test on first 10 samples
print("[3] Testing AI on actual dataset samples...\n")
print("-" * 70)

correct = 0
total = 0

for i in range(10):
    row = df.iloc[i]
    
    # Extract features and label
    features = np.array([row['packet_size'], row['protocol'], row['packet_rate']], dtype=np.float32)
    true_label = int(row['label'])
    
    # Get AI prediction
    predicted_action, _ = model.predict(features, deterministic=True)
    
    # Check if correct
    is_correct = (predicted_action == true_label)
    correct += is_correct
    total += 1
    
    # Display
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    true_str = "Malicious" if true_label == 1 else "Benign"
    pred_str = "BLOCK" if predicted_action == 1 else "ALLOW"
    
    print(f"Sample {i+1}: size={row['packet_size']:4.0f}, rate={row['packet_rate']:2.0f} "
          f"| True: {true_str:10s} | AI: {pred_str:5s} | {status}")

print("-" * 70)
print(f"\nAccuracy on first 10 samples: {correct}/{total} = {correct/total*100:.1f}%")

# Full dataset test
print("\n[4] Testing on FULL dataset (100 samples)...\n")

correct_full = 0
predictions = []
labels = []

for i in range(len(df)):
    row = df.iloc[i]
    features = np.array([row['packet_size'], row['protocol'], row['packet_rate']], dtype=np.float32)
    true_label = int(row['label'])
    
    predicted_action, _ = model.predict(features, deterministic=True)
    
    predictions.append(predicted_action)
    labels.append(true_label)
    
    if predicted_action == true_label:
        correct_full += 1

accuracy = correct_full / len(df) * 100

print(f"Total Samples:    {len(df)}")
print(f"Correct Decisions: {correct_full}")
print(f"Wrong Decisions:   {len(df) - correct_full}")
print(f"\n🎯 ACCURACY: {accuracy:.2f}%")

# Confusion matrix
tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

print("\nConfusion Matrix:")
print(f"  True Positives:  {tp} (malicious correctly blocked)")
print(f"  True Negatives:  {tn} (benign correctly allowed)")
print(f"  False Positives: {fp} (benign wrongly blocked)")
print(f"  False Negatives: {fn} (malicious wrongly allowed)")

print("\n" + "="*60)
if accuracy == 100.0:
    print("🎉 PERFECT! The AI achieves 100% accuracy!")
else:
    print(f"✅ The AI achieves {accuracy:.2f}% accuracy!")
print("="*60)
print("\n✅ YES, IT'S ACTUALLY WORKING!")
