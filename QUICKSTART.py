# QUICK START GUIDE
# ===================

# 1. SETUP (Do this once)
# -----------------------
# Create virtual environment:
#   python -m venv venv
#   venv\Scripts\activate (Windows) or source venv/bin/activate (Linux)
#
# Install dependencies:
#   pip install -r requirements.txt


# 2. STAGE 1: TRAIN THE AI (SAFE)
# --------------------------------
# Train the DQN model on traffic data
python -m firewall.train

# This will:
# - Load data/traffic.csv
# - Train for 50,000 timesteps
# - Save model to model/firewall_dqn.zip


# 3. STAGE 2: OBSERVE TRAFFIC (SAFE)
# -----------------------------------
# Watch AI make decisions without blocking
# Requires admin/root privileges

# Windows (run terminal as Administrator):
python -m runtime.sniff

# Linux:
sudo python -m runtime.sniff


# 4. EVALUATE MODEL PERFORMANCE
# ------------------------------
python -m firewall.evaluate

# This generates:
# - Performance metrics
# - Confusion matrix
# - Visualization plots


# 5. STAGE 3: DEPLOY AI FIREWALL (CAUTION)
# -----------------------------------------
# First run in OBSERVE mode (default)
python -m runtime.policy

# To enable REAL BLOCKING:
# Edit runtime/policy.py and set OBSERVE_ONLY = False
# Then run again (with admin privileges)


# 6. MANUAL FIREWALL CONTROL
# ---------------------------
# Block an IP:
python -m runtime.firewall_controller --block 192.168.1.100

# Unblock an IP:
python -m runtime.firewall_controller --unblock 192.168.1.100

# Unblock all:
python -m runtime.firewall_controller --unblock-all

# List blocked IPs:
python -m runtime.firewall_controller --list


# 7. TROUBLESHOOTING
# ------------------
# If you get "Permission Denied":
#   - Run with administrator/root privileges
#
# If model not found:
#   - Run training first: python -m firewall.train
#
# If Scapy errors on Windows:
#   - Install Npcap: https://npcap.com/


# 8. ADVANCED: CUSTOM TRAINING
# -----------------------------
# To train with your own data:
# 1. Replace data/traffic.csv with your dataset
#    Format: packet_size,protocol,packet_rate,label
# 2. Run: python -m firewall.train


# 9. PROJECT STRUCTURE
# ---------------------
# adaptive-firewall-ai/
# ├── data/                    # Training datasets
# ├── firewall/                # RL training code
# ├── runtime/                 # Live deployment code
# ├── model/                   # Saved models
# ├── requirements.txt         # Dependencies
# └── README.md                # Full documentation


# 10. SAFETY REMINDERS
# --------------------
# ⚠️  ALWAYS test in OBSERVE mode first
# ⚠️  Verify model performance before live deployment
# ⚠️  Only use on networks you own/have authorization for
# ⚠️  Keep unblock-all command ready as backup


# For detailed documentation, see README.md
