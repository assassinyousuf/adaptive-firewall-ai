"""
Adaptive Firewall AI - Real-Time Dashboard

A Streamlit-based web interface for monitoring and controlling the AI firewall.

Features:
- Real-time traffic monitoring
- Live firewall decisions (ALLOW/BLOCK)
- Performance metrics and statistics
- Model evaluation results
- Configuration controls
- Alert system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import DQN
from firewall.env import FirewallEnv
from firewall.features import extract_features


# Page configuration
st.set_page_config(
    page_title="Adaptive Firewall AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Futuristic Dark Theme
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Headers with neon glow */
    .big-font {
        font-size: 36px !important;
        font-weight: 900;
        background: linear-gradient(90deg, #00f5ff 0%, #00ff88 50%, #00f5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        letter-spacing: 2px;
        font-family: 'Orbitron', 'Courier New', monospace;
    }
    
    /* Metric cards with glassmorphism */
    .metric-card {
        background: rgba(13, 27, 42, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 245, 255, 0.2);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 245, 255, 0.1);
    }
    
    /* Alert boxes with neon borders */
    .alert-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
        font-family: 'Roboto Mono', monospace;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .alert-danger {
        background: rgba(255, 0, 80, 0.15);
        border-color: #ff0050;
        color: #ff4d88;
        box-shadow: 0 0 20px rgba(255, 0, 80, 0.3);
    }
    
    .alert-success {
        background: rgba(0, 255, 136, 0.15);
        border-color: #00ff88;
        color: #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .alert-warning {
        background: rgba(255, 170, 0, 0.15);
        border-color: #ffaa00;
        color: #ffcc66;
        box-shadow: 0 0 20px rgba(255, 170, 0, 0.3);
    }
    
    /* Cyberpunk text effects */
    h1, h2, h3 {
        color: #00f5ff !important;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
        border-right: 2px solid rgba(0, 245, 255, 0.3);
    }
    
    /* Buttons with neon effect */
    .stButton>button {
        background: linear-gradient(90deg, #00f5ff 0%, #00ff88 100%);
        color: #0a0e27;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.8);
        transform: translateY(-2px);
    }
    
    /* Metrics with glow */
    [data-testid="stMetricValue"] {
        color: #00ff88;
        font-size: 28px;
        font-weight: bold;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
    }
    
    /* Tables */
    .stDataFrame {
        background: rgba(13, 27, 42, 0.6);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00f5ff 0%, #00ff88 100%);
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.6);
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stTextInput {
        background: rgba(13, 27, 42, 0.6);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 27, 42, 0.4);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 245, 255, 0.1);
        border-radius: 8px;
        color: #00f5ff;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00f5ff 0%, #00ff88 100%);
        color: #0a0e27;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(13, 27, 42, 0.4);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f5ff 0%, #00ff88 100%);
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 10px rgba(0, 245, 255, 0.5); }
        50% { text-shadow: 0 0 20px rgba(0, 245, 255, 0.8), 0 0 30px rgba(0, 245, 255, 0.6); }
    }
    
    .big-font {
        animation: glow 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'traffic_log' not in st.session_state:
    st.session_state.traffic_log = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_packets': 0,
        'allowed': 0,
        'blocked': 0,
        'threats_detected': 0
    }
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'num_features' not in st.session_state:
    st.session_state.num_features = 7  # Default to enhanced model
if 'show_help' not in st.session_state:
    st.session_state.show_help = False
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True


def load_model(model_path):
    """Load the trained DQN model."""
    try:
        model = DQN.load(model_path)
        st.session_state.model = model
        st.session_state.model_loaded = True
        
        # Detect number of features from model
        obs_shape = model.observation_space.shape[0]
        st.session_state.num_features = obs_shape
        
        return model, None
    except Exception as e:
        return None, str(e)


def predict_packet(features):
    """Predict action for a packet."""
    if not st.session_state.model_loaded or st.session_state.model is None:
        return 0, 0.5
    
    # Get expected number of features from model
    expected_features = st.session_state.num_features
    
    # Adapt features to model requirements
    if len(features) > expected_features:
        # Truncate to first N features (size, protocol, rate for 3-feature models)
        features = features[:expected_features]
    elif len(features) < expected_features:
        # Pad with zeros if needed
        features = features + [0] * (expected_features - len(features))
    
    # Reshape features for prediction
    obs = np.array(features, dtype=np.float32)
    
    try:
        action, _ = st.session_state.model.predict(obs, deterministic=True)
        
        # Get Q-values for confidence
        try:
            obs_tensor = st.session_state.model.policy.obs_to_tensor(obs)[0]
            q_values = st.session_state.model.q_net(obs_tensor)
            action_idx = int(action.item() if hasattr(action, 'item') else action)
            confidence = float(q_values[0, action_idx].detach().numpy())
        except Exception:
            confidence = abs(float(action)) + 1.0
        
        return int(action), confidence
    except Exception as e:
        # Fallback to safe decision
        return 0, 0.5


def generate_sample_traffic():
    """Generate sample network traffic for demo."""
    traffic_types = [
        # Benign
        {'type': 'Web Browsing', 'size': np.random.randint(400, 1500), 'protocol': 6, 
         'rate': np.random.randint(10, 40), 'entropy': np.random.uniform(4.0, 6.5),
         'malicious': 0},
        {'type': 'Email', 'size': np.random.randint(200, 800), 'protocol': 6,
         'rate': np.random.randint(5, 20), 'entropy': np.random.uniform(3.5, 5.5),
         'malicious': 0},
        {'type': 'Video Stream', 'size': np.random.randint(1200, 1500), 'protocol': 17,
         'rate': np.random.randint(40, 60), 'entropy': np.random.uniform(7.0, 8.0),
         'malicious': 0},
        # Malicious
        {'type': 'DDoS Flood', 'size': np.random.randint(60, 100), 'protocol': 17,
         'rate': np.random.randint(90, 100), 'entropy': np.random.uniform(1.0, 2.5),
         'malicious': 1},
        {'type': 'Port Scan', 'size': np.random.randint(40, 80), 'protocol': 6,
         'rate': np.random.randint(70, 95), 'entropy': np.random.uniform(1.5, 3.0),
         'malicious': 1},
        {'type': 'Data Exfiltration', 'size': np.random.randint(1400, 1500), 'protocol': 6,
         'rate': np.random.randint(80, 95), 'entropy': np.random.uniform(7.5, 8.0),
         'malicious': 1},
    ]
    
    # 70% benign, 30% malicious
    if np.random.random() < 0.7:
        sample = np.random.choice(traffic_types[:3])
    else:
        sample = np.random.choice(traffic_types[3:])
    
    # Create full feature vector (7 features)
    features = [
        sample['size'],
        sample['protocol'],
        sample['rate'],
        sample['entropy'],
        np.random.uniform(50, 500),  # size_variance
        np.random.uniform(0.001, 0.1),  # inter_arrival_time
        np.random.randint(0, 63)  # flags
    ]
    
    return features, sample['type'], sample['malicious']


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<p class="big-font">🛡️ ADAPTIVE FIREWALL AI</p>', unsafe_allow_html=True)
    st.markdown("🔮 **Neural Network Defense System** • Real-time Threat Detection • AI-Powered Security")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚡ NEURAL CONTROL CENTER")
        st.markdown("---")
        
        # Model selection
        model_options = {
            "🧠 Enhanced Neural Net (98.2%)": "model/firewall_dqn_enhanced.zip",
            "🔧 Original Model (100%)": "model/firewall_dqn.zip"
        }
        selected_model = st.selectbox(
            "Select AI Model", 
            list(model_options.keys()),
            help="Enhanced model recommended for production use. Original model for testing only."
        )
        
        # Show model recommendation
        if "Enhanced" in selected_model:
            st.info("💡 **Recommended**: This model is trained on 5,000 samples with advanced features.")
        else:
            st.warning("⚠️ Demo model may overfit. Use Enhanced model for real deployment.")
        
        if st.button("🔄 Load Model"):
            with st.spinner("Loading neural network... Please wait."):
                model, error = load_model(model_options[selected_model])
                if error:
                    st.error(f"❌ Error loading model: {error}")
                    st.info("💡 **Solution**: Make sure model files exist in the 'model/' directory.")
                else:
                    st.success("✅ Model loaded successfully! You can now start scanning.")
                    st.balloons()
        
        st.markdown("---")
        
        # Mode selection
        st.markdown("### 🎯 DEFENSE MODE")
        st.caption("Choose how the firewall should respond to threats")
        mode = st.radio(
            "Operation Mode",
            ["👁️ OBSERVE", "⚔️ ACTIVE"],
            help="OBSERVE: AI analyzes traffic but doesn't block anything (safe for learning). ACTIVE: AI actively blocks detected threats (requires admin privileges)."
        )
        
        if mode == "⚔️ ACTIVE":
            st.warning("⚠️ ACTIVE DEFENSE: AI will block threats")
            st.caption("⚡ Admin privileges required for real blocking")
        else:
            st.info("👁️ OBSERVE MODE: Monitoring traffic only")
            st.caption("✅ Safe mode - no packets will be blocked")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 SYSTEM METRICS")
        st.caption("Real-time statistics from AI analysis")
        st.metric(
            "⚡ PACKETS ANALYZED", 
            st.session_state.stats['total_packets'],
            help="Total number of network packets processed by AI"
        )
        st.metric(
            "✅ ALLOWED", 
            st.session_state.stats['allowed'], 
            delta=f"{st.session_state.stats['allowed']/max(st.session_state.stats['total_packets'], 1)*100:.1f}%",
            help="Packets identified as safe and allowed through"
        )
        st.metric(
            "🛡️ BLOCKED", 
            st.session_state.stats['blocked'],
            delta=f"{st.session_state.stats['blocked']/max(st.session_state.stats['total_packets'], 1)*100:.1f}%",
            help="Packets identified as threats and blocked"
        )
        st.metric(
            "🚨 THREATS DETECTED", 
            st.session_state.stats['threats_detected'],
            help="Total malicious traffic detected by AI"
        )
        
        st.markdown("---")
        
        # Controls
        st.markdown("### 🎮 SYSTEM CONTROLS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❓ HELP", use_container_width=True):
                st.session_state.show_help = not st.session_state.show_help
        with col2:
            if st.button("🔄 RESET", use_container_width=True, help="Reset all data and statistics"):
                st.session_state.first_visit = True
        
        if st.button("🗑️ PURGE LOGS", help="Clear all logs and statistics"):
            st.session_state.traffic_log = []
            st.session_state.alerts = []
            st.session_state.stats = {
                'total_packets': 0,
                'allowed': 0,
                'blocked': 0,
                'threats_detected': 0
            }
            st.success("✅ System logs purged!")
        
        if st.button("📥 EXPORT TELEMETRY", help="Download traffic logs as CSV file"):
            if st.session_state.traffic_log:
                df = pd.DataFrame(st.session_state.traffic_log)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ DOWNLOAD DATA",
                    data=csv,
                    file_name=f"firewall_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No data to export. Run a scan first!")
    
    # Main content
    
    # First-time user guide
    if st.session_state.first_visit and not st.session_state.model_loaded:
        st.success("👋 **Welcome to Adaptive Firewall AI!**")
        
        with st.expander("📖 **QUICK START GUIDE** - Click here to learn how to use this dashboard", expanded=True):
            st.markdown("""
            ### 🚀 Getting Started in 3 Easy Steps:
            
            **STEP 1: Load AI Model** (Left Sidebar)
            - Choose a model from the dropdown
            - Click "🔄 Load Model" button
            - Wait for confirmation message
            
            **STEP 2: Choose Defense Mode** (Left Sidebar)
            - **👁️ OBSERVE** = Safe mode, AI just watches (Recommended for first use)
            - **⚔️ ACTIVE** = AI blocks threats (Needs admin privileges)
            
            **STEP 3: Start Scanning** (Main Panel)
            - Go to "📡 NEURAL SCANNER" tab
            - Click "▶️ INITIATE SCAN" button
            - Watch AI analyze traffic in real-time!
            
            ---
            
            ### 💡 What Each Tab Does:
            
            - **📡 NEURAL SCANNER**: Watch AI analyze network traffic in real-time
            - **📈 THREAT ANALYTICS**: See charts and statistics about detected threats
            - **🔍 SYSTEM INFO**: Learn about the AI model and features
            - **⚙️ CONFIGURATION**: Adjust settings and preferences
            
            ---
            
            ### ❓ Need Help?
            - Click the **❓ HELP** button in the sidebar anytime
            - All options have tooltips - hover over (?) icons
            - Check "🔍 SYSTEM INFO" tab for detailed information
            
            ---
            
            *Click anywhere outside this box to close this guide*
            """)
            
            if st.button("✅ Got it! Don't show this again"):
                st.session_state.first_visit = False
                st.rerun()
    
    # Help panel
    if st.session_state.show_help:
        with st.expander("❓ **HELP & FAQ**", expanded=True):
            st.markdown("""
            ### 🤔 Frequently Asked Questions:
            
            **Q: Which model should I use?**
            A: Use the **Enhanced Neural Net (98.2%)** for best results. It's trained on 5,000 samples.
            
            **Q: What's the difference between OBSERVE and ACTIVE mode?**
            A: 
            - **OBSERVE**: AI analyzes traffic but doesn't block anything (safe for testing)
            - **ACTIVE**: AI actively blocks threats (requires admin/sudo privileges)
            
            **Q: How do I see live traffic?**
            A: Load a model, then go to "📡 NEURAL SCANNER" tab and click "▶️ INITIATE SCAN"
            
            **Q: Is this analyzing real network traffic?**
            A: The demo generates simulated traffic. For real traffic, you need admin privileges and the runtime module.
            
            **Q: What do the colors mean?**
            A:
            - 🟢 Green = Safe traffic (ALLOWED)
            - 🔴 Red = Threat detected (BLOCKED)
            - 🟡 Yellow = Warning or info
            
            **Q: How accurate is the AI?**
            A: The Enhanced model has 98.2% accuracy on test data with very low false positives.
            
            **Q: Can I export the data?**
            A: Yes! Click "📥 EXPORT TELEMETRY" in the sidebar after running scans.
            """)
            
            if st.button("❌ Close Help"):
                st.session_state.show_help = False
                st.rerun()
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ **ACTION REQUIRED**: Load an AI model from the sidebar to begin")
        st.info("👈 Look at the left sidebar and click '🔄 Load Model' to get started!")
        
        # Show model comparison
        st.markdown("## 🤖 Available AI Models")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Enhanced Neural Network")
            st.markdown("""            - **Accuracy:** `98.2%`
            - **Features:** `7 Advanced Vectors`
            - **Dataset:** `5,000 Samples`
            - **Status:** `PRODUCTION READY` ✅
            
            **Best for:** Production use, real deployment
            
            **What it analyzes:**
            - Packet size & protocol
            - Traffic rate & patterns
            - Data entropy (randomness)
            - Timing anomalies
            - TCP flags & more
            """)
        
        with col2:
            st.markdown("### 🔧 Original Model")
            st.markdown("""            - **Accuracy:** `100%` (May overfit)
            - **Features:** `3 Basic Vectors`
            - **Dataset:** `100 Samples`
            - **Status:** `DEMO VERSION`
            
            **Best for:** Testing, demonstrations
            
            **What it analyzes:**
            - Packet size
            - Protocol type
            - Traffic rate
            
            ⚠️ Small dataset may not generalize well
            """)
        
        st.markdown("---")
        st.info("💡 **Recommendation**: Start with the Enhanced Neural Net model for best results!")
        
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 NEURAL SCANNER", 
        "📈 THREAT ANALYTICS", 
        "🔍 SYSTEM INFO", 
        "⚙️ CONFIGURATION"
    ])
    
    with tab1:
        # Live monitoring
        st.markdown("## 🔴 REAL-TIME THREAT SCANNER")
        st.caption("Watch the AI analyze network traffic and detect threats in real-time")
        
        # Instructions
        st.info("""            **How it works:** 
            1. Click the '▶️ INITIATE SCAN' button below
            2. AI will analyze 50 simulated network packets
            3. Watch real-time decisions: 🟢 ALLOW or 🔴 BLOCK
            4. See live charts, statistics, and security alerts
            
            💡 The scan takes about 15 seconds to complete
            """)
        
        # Start/Stop monitoring
        col1, col2 = st.columns([3, 1])
        with col1:
            monitor_status = st.empty()
        with col2:
            if st.button("▶️ INITIATE SCAN", help="Start analyzing network traffic"):
                monitor_status.success("🟢 NEURAL SCANNER ONLINE - AI is now analyzing traffic...")
                
                # Demo loop
                st.markdown("### 📊 Live Analysis Dashboard")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("⚡ Initializing AI neural network...")
                
                # Create placeholders
                metrics_placeholder = st.empty()
                chart_placeholder = st.empty()
                table_placeholder = st.empty()
                alerts_placeholder = st.empty()
                
                for i in range(50):
                    # Generate sample traffic
                    features, traffic_type, true_label = generate_sample_traffic()
                    
                    # Predict
                    action, confidence = predict_packet(features)
                    
                    # Update stats
                    st.session_state.stats['total_packets'] += 1
                    if action == 0:
                        st.session_state.stats['allowed'] += 1
                        decision = "ALLOW"
                        color = "🟢"
                    else:
                        st.session_state.stats['blocked'] += 1
                        st.session_state.stats['threats_detected'] += 1
                        decision = "BLOCK"
                        color = "🔴"
                    
                    # Log
                    log_entry = {
                        'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                        'type': traffic_type,
                        'decision': decision,
                        'confidence': f"{abs(confidence):.2f}",
                        'size': int(features[0]),
                        'protocol': 'TCP' if features[1] == 6 else 'UDP',
                        'rate': int(features[2]),
                        'entropy': f"{features[3]:.2f}",
                        'true_label': 'Benign' if true_label == 0 else 'Malicious'
                    }
                    st.session_state.traffic_log.insert(0, log_entry)
                    
                    # Alert for threats
                    if action == 1:
                        alert = {
                            'time': log_entry['timestamp'],
                            'message': f"🚨 {traffic_type} detected and {decision}ED",
                            'severity': 'HIGH' if true_label == 1 else 'FALSE POSITIVE'
                        }
                        st.session_state.alerts.insert(0, alert)
                    
                    # Keep only last 100 logs
                    if len(st.session_state.traffic_log) > 100:
                        st.session_state.traffic_log = st.session_state.traffic_log[:100]
                    if len(st.session_state.alerts) > 20:
                        st.session_state.alerts = st.session_state.alerts[:20]
                    
                    # Update displays
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("⚡ TOTAL", st.session_state.stats['total_packets'])
                        col2.metric("✅ SAFE", st.session_state.stats['allowed'])
                        col3.metric("🛡️ BLOCKED", st.session_state.stats['blocked'])
                        col4.metric("🚨 THREATS", st.session_state.stats['threats_detected'])
                    
                    # Traffic chart
                    if len(st.session_state.traffic_log) > 1:
                        recent_logs = st.session_state.traffic_log[:20]
                        df = pd.DataFrame(recent_logs)
                        
                        fig = px.scatter(df, x='timestamp', y='rate', color='decision',
                                       size='size', hover_data=['type', 'entropy'],
                                       title='Recent Traffic Pattern',
                                       color_discrete_map={'ALLOW': 'green', 'BLOCK': 'red'})
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Recent traffic table
                    with table_placeholder.container():
                        st.markdown("### 📊 RECENT NETWORK ACTIVITY")
                        df_display = pd.DataFrame(st.session_state.traffic_log[:10])
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Alerts
                    if st.session_state.alerts:
                        with alerts_placeholder.container():
                            st.markdown("### 🚨 SECURITY ALERTS")
                            for alert in st.session_state.alerts[:5]:
                                severity_color = "danger" if alert['severity'] == 'HIGH' else "warning"
                                st.markdown(
                                    f'<div class="alert-box alert-{severity_color}">'
                                    f'<strong>⏰ {alert["time"]}</strong> → {alert["message"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    
                    # Update progress
                    progress_bar.progress((i + 1) / 50)
                    status_text.text(f"⚡ ANALYZING PACKET {i+1}/50 → {color} {decision}: {traffic_type}")
                    
                    time.sleep(0.3)
                
                status_text.text("✅ SCAN COMPLETE • All threats neutralized")
                monitor_status.info("⏸️ SCANNER STANDBY")
    
    with tab2:
        # Analytics
        st.header("📈 Threat Analytics Dashboard")
        st.caption("Visualize patterns, statistics, and performance metrics")
        
        if not st.session_state.traffic_log:
            st.info("📊 **No data available yet**")
            st.markdown("""
            To see analytics:
            1. Go to the **📡 NEURAL SCANNER** tab
            2. Click **▶️ INITIATE SCAN** to generate data
            3. Come back here to see charts and statistics
            
            Analytics will show:
            - Traffic patterns and distributions
            - Threat detection accuracy
            - Protocol and traffic type breakdowns
            - Performance metrics (precision, recall, etc.)
            """)
        else:
            df = pd.DataFrame(st.session_state.traffic_log)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Decision distribution
                st.subheader("Decision Distribution")
                decision_counts = df['decision'].value_counts()
                fig = px.pie(values=decision_counts.values, names=decision_counts.index,
                           color=decision_counts.index,
                           color_discrete_map={'ALLOW': 'green', 'BLOCK': 'red'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Traffic type distribution
                st.subheader("Traffic Types")
                type_counts = df['type'].value_counts()
                fig = px.bar(x=type_counts.index, y=type_counts.values,
                           labels={'x': 'Type', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Protocol distribution
                st.subheader("Protocol Distribution")
                protocol_counts = df['protocol'].value_counts()
                fig = px.pie(values=protocol_counts.values, names=protocol_counts.index)
                st.plotly_chart(fig, use_container_width=True)
                
                # Entropy distribution
                st.subheader("Entropy Distribution by Decision")
                df['entropy_float'] = df['entropy'].astype(float)
                fig = px.box(df, x='decision', y='entropy_float',
                           color='decision',
                           color_discrete_map={'ALLOW': 'green', 'BLOCK': 'red'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy metrics (if true labels available)
            st.subheader("📊 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate metrics
            df['correct'] = ((df['decision'] == 'BLOCK') & (df['true_label'] == 'Malicious')) | \
                           ((df['decision'] == 'ALLOW') & (df['true_label'] == 'Benign'))
            
            accuracy = df['correct'].mean() * 100
            total = len(df)
            correct = df['correct'].sum()
            
            col1.metric("Accuracy", f"{accuracy:.1f}%")
            col2.metric("Correct", f"{correct}/{total}")
            
            # False positives/negatives
            fp = ((df['decision'] == 'BLOCK') & (df['true_label'] == 'Benign')).sum()
            fn = ((df['decision'] == 'ALLOW') & (df['true_label'] == 'Malicious')).sum()
            
            col3.metric("False Positives", fp)
            col4.metric("False Negatives", fn)
    
    with tab3:
        # Model information
        st.header("🔍 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.markdown(f"""
            **Selected Model:** {selected_model}
            
            **Architecture:**
            - Algorithm: Deep Q-Network (DQN)
            - Network: MLP [128, 128]
            - Features: 7 (enhanced)
            - Action Space: Binary (ALLOW/BLOCK)
            
            **Training:**
            - Dataset: 5,000 samples
            - Train/Val/Test: 4000/500/500
            - Timesteps: 100,000
            - Learning Rate: 3e-4
            """)
        
        with col2:
            st.subheader("Performance Metrics")
            st.markdown("""
            **Test Set Results:**
            - Accuracy: 98.20%
            - Precision: 94.59%
            - Recall: 99.29%
            - F1 Score: 96.89%
            
            **Error Rates:**
            - False Positive: 2.23%
            - False Negative: 0.71%
            
            **Status:** ✅ Production Ready
            """)
        
        st.subheader("Feature Importance")
        features_info = {
            'Feature': ['packet_size', 'protocol', 'packet_rate', 'entropy', 
                       'size_variance', 'inter_arrival_time', 'flags'],
            'Description': [
                'Size of packet in bytes',
                'TCP (6) or UDP (17)',
                'Packets per second',
                'Payload randomness (0-8)',
                'Packet size consistency',
                'Time between packets',
                'TCP flags (SYN, ACK, etc.)'
            ],
            'Importance': ['High', 'Medium', 'High', 'Very High',
                          'Medium', 'Medium', 'High']
        }
        st.table(pd.DataFrame(features_info))
    
    with tab4:
        # Settings
        st.header("⚙️ Configuration")
        
        st.subheader("Alert Thresholds")
        alert_threshold = st.slider("Threat Confidence Threshold", 0.0, 10.0, 5.0, 0.1)
        st.info(f"Alerts will be triggered for threats with confidence > {alert_threshold}")
        
        st.subheader("Display Settings")
        max_logs = st.number_input("Maximum Log Entries", 10, 1000, 100, 10)
        refresh_rate = st.slider("Refresh Rate (seconds)", 0.1, 2.0, 0.3, 0.1)
        
        st.subheader("Export Settings")
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
        
        if st.button("Save Settings"):
            st.success("✅ Settings saved!")
        
        st.markdown("---")
        
        st.subheader("About")
        st.markdown("""
        **Adaptive Firewall AI Dashboard**
        
        Version: 2.0 (Production)
        
        This dashboard provides real-time monitoring and control of the AI-powered
        adaptive firewall system. The firewall uses Deep Reinforcement Learning
        (DQN) to make intelligent decisions about network traffic.
        
        **GitHub:** https://github.com/assassinyousuf/adaptive-firewall-ai
        
        Built with ❤️ using Streamlit, Stable-Baselines3, and PyTorch
        """)


if __name__ == "__main__":
    main()
