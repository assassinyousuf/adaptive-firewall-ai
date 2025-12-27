# Adaptive Firewall AI

🧠 **An AI-powered adaptive firewall using Reinforcement Learning to automatically classify and block malicious network traffic without predefined rules.**

Built with Python, Scapy, and Stable-Baselines3 (DQN).

---

## 📋 Project Overview

This system uses **Deep Q-Learning (DQN)** to learn traffic patterns and adaptively decide whether to allow or block network packets. Unlike traditional firewalls with static rules, this AI firewall **learns** from traffic behavior.

### Key Features
- ✅ Real-time packet capture using Scapy
- ✅ Reinforcement Learning (DQN) for decision-making
- ✅ Feature extraction from network packets
- ✅ Safe "observe-only" mode for testing
- ✅ Cross-platform firewall control (Linux iptables / Windows Firewall)
- ✅ Model evaluation and performance metrics
- ✅ Research-ready architecture

---

## 🏗️ Project Structure

```
adaptive-firewall-ai/
│
├── data/
│   └── traffic.csv              # Training dataset
│
├── firewall/                    # STAGE 1: Offline RL Training
│   ├── __init__.py
│   ├── env.py                   # RL Environment (Gym)
│   ├── features.py              # Feature extraction
│   ├── rewards.py               # Reward function
│   ├── train.py                 # Model training
│   └── evaluate.py              # Model evaluation
│
├── runtime/                     # STAGE 2 & 3: Live Deployment
│   ├── __init__.py
│   ├── sniff.py                 # Packet capture (observe mode)
│   ├── policy.py                # AI decision engine
│   └── firewall_controller.py   # System firewall integration
│
├── model/                       # Trained models
│   └── firewall_dqn.zip         # (generated after training)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Windows (with admin privileges) or Linux (with root access)
- pip

### Setup

1. **Clone/Create Project**
   ```bash
   mkdir adaptive-firewall-ai
   cd adaptive-firewall-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📚 Three-Stage Workflow

### ⚡ STAGE 1: Offline Training (SAFE)

Train the AI on historical/simulated traffic data.

```bash
# Train the DQN model
python -m firewall.train
```

This creates `model/firewall_dqn.zip` containing the trained agent.

**What happens:**
- Loads `data/traffic.csv`
- Trains DQN for 50,000 timesteps
- Saves model to `model/` directory
- No network interaction

---

### 🔍 STAGE 2: Live Observation (SAFE)

Capture real packets and watch AI decisions **without blocking**.

```bash
# Observe packets (requires admin/root)
sudo python -m runtime.sniff     # Linux
python -m runtime.sniff           # Windows (run as admin)
```

**What happens:**
- Captures 50 packets
- Extracts features
- Displays packet information
- No blocking occurs

---

### 🛡️ STAGE 3: Active Firewall (CAUTION)

Deploy AI with real blocking capability.

```bash
# Run in observe-only mode first
python -m runtime.policy
```

**Safety Settings** (in `runtime/policy.py`):
```python
OBSERVE_ONLY = True   # ⚠️ Set to False to enable real blocking
PACKET_COUNT = 100    # Number of packets to process
```

**What happens:**
- AI processes live traffic
- Makes allow/block decisions
- If `OBSERVE_ONLY=False`: executes iptables/firewall rules

---

## 📊 Evaluation

Evaluate model performance:

```bash
python -m firewall.evaluate
```

**Outputs:**
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix
- Results saved to `model/evaluation_report.txt`
- Visualization in `model/evaluation_results.png`

---

## 🔧 Configuration

### Modify Training Parameters

Edit `firewall/train.py`:
```python
total_timesteps = 50000      # Training duration
learning_rate = 1e-3         # DQN learning rate
buffer_size = 50000          # Experience replay buffer
```

### Modify Features

Edit `firewall/features.py` to add more features:
```python
def extract_features(row):
    return [
        row["packet_size"],
        row["protocol"],
        row["packet_rate"],
        # Add more features here
    ]
```

### Modify Reward Function

Edit `firewall/rewards.py` to change learning behavior:
```python
def calculate_reward(action, label):
    if action == label:
        return +5.0  # Correct decision
    return -10.0     # Incorrect decision
```

---

## 🎯 Usage Examples

### Train with Custom Dataset
```bash
# Replace data/traffic.csv with your dataset
python -m firewall.train
```

### Evaluate Model
```bash
python -m firewall.evaluate
```

### Manual Firewall Control
```bash
# Block specific IP
python -m runtime.firewall_controller --block 192.168.1.100

# Unblock IP
python -m runtime.firewall_controller --unblock 192.168.1.100

# Unblock all
python -m runtime.firewall_controller --unblock-all

# List blocked IPs
python -m runtime.firewall_controller --list
```

---

## ⚠️ Safety & Ethics

### Before Deployment
1. ✅ Test extensively in **OBSERVE mode**
2. ✅ Verify model performance on evaluation set
3. ✅ Understand False Positive/Negative rates
4. ✅ Have rollback plan (unblock all script)
5. ✅ Get proper authorization for network monitoring

### Limitations
- Trained on limited dataset (enhance with real data)
- May have false positives/negatives
- Requires continuous learning for new threats
- Performance depends on feature quality

---

## 🔬 Research & Academic Use

This project is structured for **research publications**:

### Suitable For:
- Bachelor/Master thesis
- Conference papers (security, ML, networking)
- Course projects
- Portfolio demonstrations

### Key Contributions:
1. Novel application of RL to firewall systems
2. Adaptive threat detection without signature matching
3. Comparison with rule-based systems
4. Performance analysis (precision, recall, F1)

### Suggested Experiments:
- Compare DQN vs PPO vs other RL algorithms
- Evaluate on public datasets (CIC-IDS2017, NSL-KDD)
- Measure adaptation speed to new threats
- Analyze feature importance
- Test transfer learning across networks

---

## 📈 Performance Metrics

After training and evaluation, you'll get:

```
Accuracy:  0.9250 (92.50%)
Precision: 0.9100
Recall:    0.8950
F1 Score:  0.9024

Confusion Matrix:
  True Positives:  45
  False Positives: 5
  True Negatives:  48
  False Negatives: 2
```

---

## 🛠️ Troubleshooting

### Issue: Permission Denied (Packet Capture)
**Solution:** Run with administrator/root privileges
```bash
sudo python -m runtime.sniff    # Linux
# Right-click > Run as Administrator (Windows)
```

### Issue: Model Not Found
**Solution:** Train the model first
```bash
python -m firewall.train
```

### Issue: Import Errors
**Solution:** Ensure virtual environment is activated and dependencies installed
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Scapy Not Capturing Packets
**Solution (Windows):** Install Npcap
- Download: https://npcap.com/
- Install with "WinPcap compatibility mode"

---

## 🚧 Roadmap / Future Enhancements

- [ ] Add more sophisticated features (packet entropy, timing patterns)
- [ ] Integrate with real IDS datasets (CIC-IDS2017, NSL-KDD)
- [ ] Implement continuous learning pipeline
- [ ] Add web dashboard for monitoring
- [ ] Support for distributed deployment
- [ ] Multi-agent coordination
- [ ] Integration with SIEM systems
- [ ] Explainable AI for decisions

---

## 📄 License

This project is for **educational and research purposes**.

⚠️ Use responsibly and only on networks you own or have authorization to monitor.

---

## 🙏 Acknowledgments

Built using:
- [Scapy](https://scapy.net/) - Packet manipulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [PyTorch](https://pytorch.org/) - Deep learning backend

---

## 📧 Contact & Support

For questions, issues, or research collaboration:
- Open an issue on GitHub
- Refer to documentation in code comments

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@software{adaptive_firewall_ai,
  title={Adaptive Firewall AI: Reinforcement Learning for Network Security},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/adaptive-firewall-ai}
}
```

---

**Happy Learning! 🚀**

Remember: Start with STAGE 1 (training), then STAGE 2 (observation), and only after thorough testing proceed to STAGE 3 (active blocking).
