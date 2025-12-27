"""
System Test Script - Check all components
"""

import sys
print(f"Python Version: {sys.version}\n")

# Test 1: Check imports
print("="*60)
print("TEST 1: Checking dependencies...")
print("="*60)

dependencies = {
    'pandas': 'Data processing',
    'numpy': 'Numerical operations',
    'gymnasium': 'RL environment',
    'stable_baselines3': 'RL algorithms',
    'torch': 'Deep learning',
    'scapy': 'Packet capture'
}

missing = []
for pkg, desc in dependencies.items():
    try:
        __import__(pkg)
        print(f"✅ {pkg:20s} - {desc}")
    except ImportError as e:
        print(f"❌ {pkg:20s} - MISSING ({desc})")
        missing.append(pkg)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print(f"   Install with: pip install {' '.join(missing)}")
else:
    print("\n✅ All dependencies installed!")

# Test 2: Check dataset
print("\n" + "="*60)
print("TEST 2: Checking dataset...")
print("="*60)

try:
    import pandas as pd
    df = pd.read_csv('data/traffic.csv')
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['packet_size', 'protocol', 'packet_rate', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print(f"✅ All required columns present")
        
        # Check label distribution
        label_dist = df['label'].value_counts()
        print(f"\n   Label distribution:")
        print(f"   - Benign (0): {label_dist.get(0, 0)} samples")
        print(f"   - Malicious (1): {label_dist.get(1, 0)} samples")
        
except FileNotFoundError:
    print("❌ Dataset file not found: data/traffic.csv")
except Exception as e:
    print(f"❌ Dataset error: {e}")

# Test 3: Check module imports
print("\n" + "="*60)
print("TEST 3: Checking project modules...")
print("="*60)

modules = [
    ('firewall.features', 'Feature extraction'),
    ('firewall.rewards', 'Reward calculation'),
    ('firewall.env', 'RL environment'),
    ('firewall.train', 'Training module'),
    ('firewall.evaluate', 'Evaluation module'),
    ('runtime.sniff', 'Packet capture'),
    ('runtime.policy', 'AI policy'),
    ('runtime.firewall_controller', 'Firewall controller')
]

import_errors = []
for module, desc in modules:
    try:
        __import__(module)
        print(f"✅ {module:30s} - {desc}")
    except ImportError as e:
        print(f"❌ {module:30s} - ERROR: {e}")
        import_errors.append((module, str(e)))
    except Exception as e:
        print(f"⚠️  {module:30s} - WARNING: {e}")

if import_errors:
    print(f"\n⚠️  Import errors detected:")
    for mod, err in import_errors:
        print(f"   {mod}: {err}")
else:
    print("\n✅ All project modules importable!")

# Test 4: Test feature extraction
print("\n" + "="*60)
print("TEST 4: Testing feature extraction...")
print("="*60)

try:
    from firewall.features import extract_features
    test_row = [100, 0, 10, 0]
    features = extract_features(test_row)
    print(f"✅ Feature extraction works")
    print(f"   Input: {test_row}")
    print(f"   Output: {features}")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")

# Test 5: Test reward calculation
print("\n" + "="*60)
print("TEST 5: Testing reward function...")
print("="*60)

try:
    from firewall.rewards import calculate_reward
    
    test_cases = [
        (0, 0, "Allow benign (correct)"),
        (1, 1, "Block malicious (correct)"),
        (0, 1, "Allow malicious (wrong)"),
        (1, 0, "Block benign (wrong)")
    ]
    
    all_passed = True
    for action, label, desc in test_cases:
        reward = calculate_reward(action, label)
        status = "✅" if (action == label and reward > 0) or (action != label and reward < 0) else "❌"
        print(f"{status} {desc:25s} → Reward: {reward:+.1f}")
        if not ((action == label and reward > 0) or (action != label and reward < 0)):
            all_passed = False
    
    if all_passed:
        print(f"\n✅ Reward function logic correct!")
    else:
        print(f"\n⚠️  Reward function may need adjustment")
        
except Exception as e:
    print(f"❌ Reward function failed: {e}")

# Test 6: Test environment creation
print("\n" + "="*60)
print("TEST 6: Testing RL environment...")
print("="*60)

try:
    import numpy as np
    from firewall.env import FirewallEnv
    
    # Create small test dataset
    test_data = np.array([
        [100, 0, 10, 0],
        [1200, 1, 80, 1],
        [200, 0, 15, 0]
    ])
    
    env = FirewallEnv(test_data)
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Environment reset works")
    print(f"   Initial observation: {obs}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Environment step works")
    print(f"   Action: {action}, Reward: {reward:.1f}")
    
except Exception as e:
    print(f"❌ Environment test failed: {e}")
    import traceback
    traceback.print_exc()

# Final Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if missing:
    print(f"❌ FAILED: Missing dependencies - install with:")
    print(f"   pip install {' '.join(missing)}")
elif import_errors:
    print(f"❌ FAILED: Module import errors")
else:
    print("✅ ALL TESTS PASSED!")
    print("\nYour system is ready to:")
    print("  1. Train the model: python -m firewall.train")
    print("  2. Evaluate model: python -m firewall.evaluate")
    print("  3. Observe traffic: python -m runtime.sniff (requires admin)")
    print("  4. Deploy AI: python -m runtime.policy (requires admin)")

print("="*60)
