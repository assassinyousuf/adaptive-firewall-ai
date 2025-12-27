"""
Quick test: Train model for 1000 steps to verify training works
"""

import numpy as np
from firewall.env import FirewallEnv
from stable_baselines3 import DQN

print("="*60)
print("QUICK TRAINING TEST (1000 timesteps)")
print("="*60)

# Generate small test dataset
np.random.seed(42)
test_data = []
for i in range(50):
    is_malicious = np.random.random() < 0.3
    if is_malicious:
        packet_size = np.random.randint(800, 1500)
        protocol = np.random.randint(0, 3)
        packet_rate = np.random.randint(50, 100)
        label = 1
    else:
        packet_size = np.random.randint(64, 800)
        protocol = np.random.randint(0, 3)
        packet_rate = np.random.randint(1, 50)
        label = 0
    test_data.append([packet_size, protocol, packet_rate, label])

test_data = np.array(test_data)
print(f"\n✅ Test dataset created: {len(test_data)} samples")

# Create environment
env = FirewallEnv(test_data)
print("✅ Environment created")

# Create model
print("\n🧠 Creating DQN model...")
model = DQN("MlpPolicy", env, verbose=0)
print("✅ Model created")

# Quick training
print("\n🚀 Starting training for 1000 timesteps...")
try:
    model.learn(total_timesteps=1000, progress_bar=True)
    print("\n✅ Training completed successfully!")
    
    # Test prediction
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    print(f"\n✅ Prediction test works")
    print(f"   Observation: {obs}")
    print(f"   Predicted action: {'BLOCK' if action == 1 else 'ALLOW'}")
    
    print("\n" + "="*60)
    print("✅ TRAINING SYSTEM FULLY FUNCTIONAL")
    print("="*60)
    print("\nYou can now run the full training:")
    print("  python -m firewall.train")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
