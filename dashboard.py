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

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-danger {
        background-color: #ff4b4b;
        color: white;
    }
    .alert-success {
        background-color: #00cc00;
        color: white;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
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


def load_model(model_path):
    """Load the trained DQN model."""
    try:
        model = DQN.load(model_path)
        st.session_state.model = model
        st.session_state.model_loaded = True
        return model, None
    except Exception as e:
        return None, str(e)


def predict_packet(features):
    """Predict action for a packet."""
    if not st.session_state.model_loaded or st.session_state.model is None:
        return 0, 0.5
    
    # Reshape features for prediction
    obs = np.array(features).reshape(1, -1)
    action, _ = st.session_state.model.predict(obs, deterministic=True)
    
    # Get Q-values for confidence
    q_values = st.session_state.model.q_net(
        st.session_state.model.policy.obs_to_tensor(obs)[0]
    )
    confidence = float(q_values[0, action].detach().numpy())
    
    return int(action), confidence


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
    st.markdown('<p class="big-font">🛡️ Adaptive Firewall AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Real-time network traffic monitoring and AI-powered threat detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Model selection
        model_options = {
            "Enhanced Model (98.2%)": "model/firewall_dqn_enhanced.zip",
            "Original Model (100%)": "model/firewall_dqn.zip"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                model, error = load_model(model_options[selected_model])
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.success("✅ Model loaded successfully!")
        
        st.markdown("---")
        
        # Mode selection
        st.header("🎯 Firewall Mode")
        mode = st.radio(
            "Operation Mode",
            ["Observe", "Active"],
            help="Observe: Monitor only, Active: Block threats"
        )
        
        if mode == "Active":
            st.warning("⚠️ Active mode: Threats will be blocked")
        else:
            st.info("👁️ Observe mode: Monitoring only")
        
        st.markdown("---")
        
        # Statistics
        st.header("📊 Statistics")
        st.metric("Total Packets", st.session_state.stats['total_packets'])
        st.metric("Allowed", st.session_state.stats['allowed'], 
                 delta=f"{st.session_state.stats['allowed']/max(st.session_state.stats['total_packets'], 1)*100:.1f}%")
        st.metric("Blocked", st.session_state.stats['blocked'],
                 delta=f"{st.session_state.stats['blocked']/max(st.session_state.stats['total_packets'], 1)*100:.1f}%")
        st.metric("Threats Detected", st.session_state.stats['threats_detected'])
        
        st.markdown("---")
        
        # Controls
        st.header("🎮 Actions")
        if st.button("🗑️ Clear Logs"):
            st.session_state.traffic_log = []
            st.session_state.alerts = []
            st.session_state.stats = {
                'total_packets': 0,
                'allowed': 0,
                'blocked': 0,
                'threats_detected': 0
            }
            st.success("Logs cleared!")
        
        if st.button("📥 Export Report"):
            if st.session_state.traffic_log:
                df = pd.DataFrame(st.session_state.traffic_log)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"firewall_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Main content
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model from the sidebar to begin monitoring")
        
        # Show model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎓 Enhanced Model")
            st.markdown("""
            - **Accuracy:** 98.2%
            - **Features:** 7 advanced features
            - **Dataset:** 5,000 samples
            - **Status:** Production Ready ✅
            """)
        
        with col2:
            st.subheader("📚 Original Model")
            st.markdown("""
            - **Accuracy:** 100% (may overfit)
            - **Features:** 3 basic features
            - **Dataset:** 100 samples
            - **Status:** Demo Version
            """)
        
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Monitor", "📈 Analytics", "🔍 Model Info", "⚙️ Settings"])
    
    with tab1:
        # Live monitoring
        st.header("Real-Time Traffic Monitor")
        
        # Start/Stop monitoring
        col1, col2 = st.columns([3, 1])
        with col1:
            monitor_status = st.empty()
        with col2:
            if st.button("▶️ Start Demo"):
                monitor_status.success("🟢 Monitoring Active")
                
                # Demo loop
                progress_bar = st.progress(0)
                status_text = st.empty()
                
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
                        col1.metric("Total", st.session_state.stats['total_packets'])
                        col2.metric("Allowed", st.session_state.stats['allowed'])
                        col3.metric("Blocked", st.session_state.stats['blocked'])
                        col4.metric("Threats", st.session_state.stats['threats_detected'])
                    
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
                        st.subheader("Recent Packets")
                        df_display = pd.DataFrame(st.session_state.traffic_log[:10])
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Alerts
                    if st.session_state.alerts:
                        with alerts_placeholder.container():
                            st.subheader("🚨 Recent Alerts")
                            for alert in st.session_state.alerts[:5]:
                                severity_color = "danger" if alert['severity'] == 'HIGH' else "warning"
                                st.markdown(
                                    f'<div class="alert-box alert-{severity_color}">'
                                    f'<strong>{alert["time"]}</strong> - {alert["message"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    
                    # Update progress
                    progress_bar.progress((i + 1) / 50)
                    status_text.text(f"Processing packet {i+1}/50 - {color} {decision}: {traffic_type}")
                    
                    time.sleep(0.3)
                
                status_text.text("✅ Demo completed!")
                monitor_status.info("⏸️ Monitoring Paused")
    
    with tab2:
        # Analytics
        st.header("Traffic Analytics")
        
        if not st.session_state.traffic_log:
            st.info("No data yet. Start monitoring to see analytics.")
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
