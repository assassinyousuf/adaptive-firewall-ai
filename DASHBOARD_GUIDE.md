# 🖥️ GUI Dashboard Guide

## Quick Start

### Launch Dashboard
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

---

## Dashboard Features

### 1. 📡 Live Monitor Tab
**Real-time traffic monitoring and AI decisions**

- **Load Model:** Select and load Enhanced or Original model
- **Start Demo:** Run automated demo with 50 sample packets
- **Real-time Display:**
  - Traffic metrics (total, allowed, blocked, threats)
  - Interactive scatter plot of traffic patterns
  - Recent packets table with full details
  - Alert notifications for blocked threats

**Traffic Information:**
- Timestamp
- Traffic type (Web Browsing, Email, DDoS, Port Scan, etc.)
- AI decision (ALLOW/BLOCK)
- Confidence score
- Packet size, protocol, rate
- Entropy value
- True label (for accuracy calculation)

---

### 2. 📈 Analytics Tab
**Statistical analysis and visualizations**

- **Decision Distribution:** Pie chart showing ALLOW vs BLOCK ratio
- **Traffic Types:** Bar chart of different traffic categories
- **Protocol Distribution:** TCP vs UDP breakdown
- **Entropy Analysis:** Box plot comparing entropy by decision
- **Performance Metrics:**
  - Overall accuracy
  - Correct predictions count
  - False positives/negatives

---

### 3. 🔍 Model Info Tab
**Model architecture and performance details**

- **Model Details:**
  - Algorithm: Deep Q-Network (DQN)
  - Network architecture: [128, 128]
  - Features: 7 advanced features
  - Training details

- **Performance Metrics:**
  - Test set accuracy: 98.20%
  - Precision: 94.59%
  - Recall: 99.29%
  - F1 Score: 96.89%
  - Error rates

- **Feature Importance:** Table explaining each of the 7 features

---

### 4. ⚙️ Settings Tab
**Configuration and controls**

- **Alert Thresholds:** Set confidence threshold for alerts
- **Display Settings:** Adjust log size and refresh rate
- **Export Settings:** Choose export format (CSV/JSON/Excel)
- **About:** Version info and GitHub link

---

## Sidebar Controls

### ⚙️ Controls Section
- **Model Selection:** Choose between Enhanced (98.2%) or Original (100%)
- **Load Model Button:** Load selected model
- **Status:** Shows model loading success/failure

### 🎯 Firewall Mode
- **Observe Mode:** Monitor only, no blocking (safe)
- **Active Mode:** Actually block threats (requires admin)

### 📊 Live Statistics
- Real-time packet counters
- Percentage calculations
- Threat detection count

### 🎮 Actions
- **Clear Logs:** Reset all traffic logs and statistics
- **Export Report:** Download CSV of traffic log

---

## Usage Scenarios

### Demo/Testing Mode
1. Load the Enhanced Model
2. Set to "Observe" mode
3. Click "Start Demo"
4. Watch AI classify 50 simulated packets
5. View analytics and performance

### Live Monitoring (Advanced)
1. Load your preferred model
2. Set to "Observe" mode (safe)
3. Monitor real network traffic
4. Analyze patterns in Analytics tab
5. Switch to "Active" when confident (requires admin)

---

## Understanding the Display

### Color Coding
- 🟢 **Green (ALLOW):** Legitimate traffic
- 🔴 **Red (BLOCK):** Malicious traffic detected

### Decision Confidence
- **Higher values:** More confident in decision
- **Q-values:** Neural network's estimated action quality

### Traffic Types
**Benign:**
- Web Browsing
- Email
- File Transfer
- Video Streaming
- DNS Queries

**Malicious:**
- DDoS Flood
- Port Scan
- Brute Force
- Data Exfiltration
- Malware C&C

---

## Tips & Tricks

### Best Practices
1. **Start with Demo:** Familiarize yourself with the interface
2. **Check Analytics:** Review performance before live deployment
3. **Use Observe Mode:** Test on real traffic safely
4. **Monitor Alerts:** Watch for false positives
5. **Export Reports:** Keep logs for analysis

### Keyboard Shortcuts
- `R` - Rerun app
- `C` - Clear cache
- `Q` - Stop running script

### Performance Tips
- Limit log entries for better performance
- Export and clear logs regularly
- Use smaller refresh rates for slower systems

---

## Troubleshooting

### Dashboard Won't Start
```bash
# Check Streamlit installation
pip install streamlit plotly psutil

# Try alternative port
streamlit run dashboard.py --server.port 8502
```

### Model Not Loading
- Check model file exists: `model/firewall_dqn_enhanced.zip`
- Verify model path in dropdown
- Retrain model if corrupted

### No Data Showing
- Click "Start Demo" to generate sample data
- Clear browser cache
- Refresh page (R key)

### Slow Performance
- Reduce max log entries in Settings
- Increase refresh rate
- Close other Streamlit apps

---

## Screenshots Guide

### Expected Display

**Header:**
```
🛡️ Adaptive Firewall AI Dashboard
Real-time network traffic monitoring and AI-powered threat detection
```

**Tabs:** Live Monitor | Analytics | Model Info | Settings

**Sidebar:**
- Model selector dropdown
- Load Model button
- Firewall mode radio buttons
- Live statistics metrics
- Action buttons

**Main Display (Live Monitor):**
- 4 metric cards at top
- Interactive scatter plot
- Recent packets table
- Alert boxes at bottom

---

## Advanced Features

### Custom Packet Analysis
Modify `generate_sample_traffic()` in `dashboard.py` to analyze specific traffic patterns

### Real-time Integration
Replace demo loop with actual packet capture from `runtime/sniff.py`

### Model Comparison
Load different models and compare their decisions on same traffic

### Export Analysis
Download CSV logs and analyze in Excel or Jupyter

---

## Next Steps

After familiarizing yourself with the dashboard:

1. **Train Enhanced Model:**
   ```bash
   python -m firewall.train_enhanced
   ```

2. **Evaluate Performance:**
   ```bash
   python -m firewall.evaluate_enhanced
   ```

3. **Deploy Live:**
   ```bash
   sudo python -m runtime.sniff --observe
   ```

4. **Monitor in Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

---

**Enjoy your AI-powered firewall dashboard! 🚀**
