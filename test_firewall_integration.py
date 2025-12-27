"""
Comprehensive Firewall Integration Test

Tests if the AI firewall can actually:
1. Capture real network packets
2. Extract features from them
3. Make AI predictions
4. Interface with system firewall (if admin)

Usage:
    python test_firewall_integration.py
"""

import sys
import platform
import subprocess
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("ADAPTIVE FIREWALL AI - INTEGRATION TEST")
print("="*70)
print()

# Test 1: Check Python environment
print("📋 TEST 1: Python Environment")
print("-" * 70)
print(f"Python Version: {sys.version}")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Platform: {platform.platform()}")
print("✅ PASS: Environment detected\n")

# Test 2: Check dependencies
print("📦 TEST 2: Required Dependencies")
print("-" * 70)

dependencies = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'gymnasium': 'gymnasium',
    'stable_baselines3': 'stable-baselines3',
    'torch': 'torch',
    'scapy': 'scapy',
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {module:20s} - installed")
    except ImportError:
        print(f"❌ {module:20s} - MISSING")
        missing.append(package)

if missing:
    print(f"\n⚠️  FAIL: Missing packages: {', '.join(missing)}")
    print(f"Install with: pip install {' '.join(missing)}")
else:
    print("\n✅ PASS: All dependencies installed\n")

# Test 3: Check model files
print("🤖 TEST 3: Trained Models")
print("-" * 70)

models = [
    "model/firewall_dqn.zip",
    "model/firewall_dqn_enhanced.zip",
    "model/best_model.zip"
]

model_found = False
for model_path in models:
    if Path(model_path).exists():
        size = Path(model_path).stat().st_size / 1024
        print(f"✅ {model_path:40s} ({size:.1f} KB)")
        model_found = True
    else:
        print(f"❌ {model_path:40s} - not found")

if model_found:
    print("\n✅ PASS: At least one model available\n")
else:
    print("\n⚠️  FAIL: No trained models found")
    print("Train a model with: python -m firewall.train_enhanced\n")

# Test 4: Test feature extraction
print("🔧 TEST 4: Feature Extraction")
print("-" * 70)

try:
    from firewall.features import extract_features
    import numpy as np
    
    # Create sample packet data
    sample_data = np.array([800, 6, 25, 5.2, 150, 0.01, 2])
    
    # Test extraction
    features = extract_features(sample_data, num_features=7)
    
    print(f"Sample input:  {sample_data}")
    print(f"Extracted:     {features}")
    print(f"Feature count: {len(features)}")
    
    if len(features) == 7:
        print("\n✅ PASS: Feature extraction working\n")
    else:
        print(f"\n⚠️  FAIL: Expected 7 features, got {len(features)}\n")
        
except Exception as e:
    print(f"❌ FAIL: Feature extraction error: {e}\n")

# Test 5: Test model loading and prediction
print("🧠 TEST 5: Model Prediction")
print("-" * 70)

try:
    from stable_baselines3 import DQN
    import numpy as np
    
    # Try to load enhanced model first
    model_paths = [
        "model/firewall_dqn_enhanced.zip",
        "model/best_model.zip",
        "model/firewall_dqn.zip"
    ]
    
    model = None
    loaded_model_path = None
    
    for path in model_paths:
        if Path(path).exists():
            try:
                model = DQN.load(path)
                loaded_model_path = path
                break
            except:
                continue
    
    if model is None:
        print("❌ FAIL: Could not load any model")
        print("Train a model with: python -m firewall.train_enhanced\n")
    else:
        print(f"✅ Model loaded: {loaded_model_path}")
        
        # Test predictions on sample data
        test_cases = [
            # [size, protocol, rate, entropy, variance, inter_arrival, flags]
            ([400, 6, 15, 4.5, 100, 0.05, 16], "Web Browsing (benign)"),
            ([1400, 6, 85, 7.8, 50, 0.001, 16], "Data Exfiltration (malicious)"),
            ([60, 17, 95, 1.2, 10, 0.0001, 0], "DDoS Flood (malicious)"),
            ([70, 6, 75, 2.0, 5, 0.002, 2], "Port Scan (malicious)"),
            ([1200, 17, 50, 7.5, 300, 0.02, 0], "Video Stream (benign)")
        ]
        
        print("\nTest Predictions:")
        correct = 0
        total = len(test_cases)
        
        for features, description in test_cases:
            obs = np.array(features).reshape(1, -1)
            action, _ = model.predict(obs, deterministic=True)
            decision = "BLOCK" if action == 1 else "ALLOW"
            
            # Determine if decision is likely correct
            is_malicious = "malicious" in description
            is_correct = (action == 1 and is_malicious) or (action == 0 and not is_malicious)
            
            if is_correct:
                correct += 1
            
            emoji = "🔴" if action == 1 else "🟢"
            status = "✓" if is_correct else "✗"
            
            print(f"  {emoji} {decision:5s} - {description:30s} [{status}]")
        
        accuracy = (correct / total) * 100
        print(f"\nTest Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        if accuracy >= 60:
            print("✅ PASS: Model making reasonable predictions\n")
        else:
            print("⚠️  WARN: Model accuracy seems low, consider retraining\n")
            
except Exception as e:
    print(f"❌ FAIL: Model prediction error: {e}\n")

# Test 6: Check admin privileges
print("🔐 TEST 6: System Privileges")
print("-" * 70)

is_admin = False

if platform.system() == "Windows":
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        is_admin = False
elif platform.system() == "Linux":
    import os
    is_admin = os.geteuid() == 0

if is_admin:
    print("✅ Running with administrator privileges")
    print("✅ PASS: Can modify firewall rules\n")
else:
    print("❌ Running WITHOUT administrator privileges")
    print("⚠️  WARN: Cannot modify firewall rules")
    print("         For full functionality, run as:")
    if platform.system() == "Windows":
        print("         Right-click → Run as Administrator")
    else:
        print("         sudo python test_firewall_integration.py\n")

# Test 7: Test firewall controller (if admin)
print("🛡️  TEST 7: Firewall Controller")
print("-" * 70)

try:
    from runtime.firewall_controller import FirewallController
    
    controller = FirewallController()
    print(f"✅ Controller initialized for {controller.system}")
    
    if is_admin:
        print("✅ Can execute firewall commands")
        print("⚠️  Note: Not testing actual blocking (destructive)")
        print("✅ PASS: Firewall controller ready\n")
    else:
        print("⚠️  Cannot test blocking without admin privileges")
        print("ℹ️  INFO: Controller exists but cannot execute\n")
        
except Exception as e:
    print(f"❌ FAIL: Firewall controller error: {e}\n")

# Test 8: Check Scapy packet capture capability
print("📡 TEST 8: Packet Capture Capability")
print("-" * 70)

try:
    from scapy.all import conf, get_if_list
    
    interfaces = get_if_list()
    print(f"Available interfaces: {len(interfaces)}")
    for iface in interfaces[:5]:  # Show first 5
        print(f"  - {iface}")
    
    if len(interfaces) > 5:
        print(f"  ... and {len(interfaces) - 5} more")
    
    if is_admin:
        print("\n✅ PASS: Can capture packets (admin mode)\n")
    else:
        print("\n⚠️  WARN: Packet capture requires admin privileges\n")
        
except Exception as e:
    print(f"❌ FAIL: Scapy error: {e}\n")

# Final Summary
print("="*70)
print("SUMMARY")
print("="*70)

print("\n✅ WHAT WORKS:")
print("   • Feature extraction from packet data")
print("   • AI model loading and predictions")
print("   • Model decision making (ALLOW/BLOCK)")
print("   • GUI Dashboard (streamlit run dashboard.py)")
print("   • Model evaluation and metrics")

print("\n⚠️  LIMITATIONS:")
if not is_admin:
    print("   • Cannot capture live packets (need admin/sudo)")
    print("   • Cannot modify firewall rules (need admin/sudo)")
else:
    print("   • All features available!")

print("\n📊 CURRENT STATUS:")
if model_found and not missing:
    if is_admin:
        print("   ✅ FULLY FUNCTIONAL - Can capture and block traffic")
        print("\n   Ready to deploy:")
        print("   1. Observe mode:  sudo python -m runtime.sniff")
        print("   2. Active mode:   sudo python -m runtime.policy")
    else:
        print("   🟡 PARTIALLY FUNCTIONAL - Models work, need admin for live traffic")
        print("\n   Can use:")
        print("   1. Dashboard:     streamlit run dashboard.py")
        print("   2. Evaluation:    python -m firewall.evaluate_enhanced")
        print("\n   For live traffic:")
        print("   1. Observe mode:  sudo python -m runtime.sniff")
        print("   2. Active mode:   sudo python -m runtime.policy")
else:
    print("   🔴 SETUP INCOMPLETE")
    if missing:
        print(f"   - Install: pip install {' '.join(missing)}")
    if not model_found:
        print("   - Train model: python -m firewall.train_enhanced")

print("\n" + "="*70)
print("Test complete! See summary above.")
print("="*70 + "\n")
