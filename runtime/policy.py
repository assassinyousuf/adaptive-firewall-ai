"""
AI-powered policy enforcement module.

STAGE 3: Active mode - uses trained model to make allow/block decisions.

Usage:
    sudo python -m runtime.policy
    
Note: 
    - Requires trained model at model/firewall_dqn.zip
    - Run in OBSERVE mode first before enabling blocking
"""

from scapy.all import sniff, IP, TCP, UDP
from stable_baselines3 import DQN
from firewall.features import extract_features_from_packet
from runtime.firewall_controller import block_ip, unblock_ip, get_blocked_ips
import time
import sys


class AdaptiveFirewall:
    """AI-powered adaptive firewall using reinforcement learning."""
    
    def __init__(self, model_path="model/firewall_dqn", observe_only=True):
        """
        Initialize adaptive firewall.
        
        Args:
            model_path: Path to trained RL model
            observe_only: If True, only log decisions without blocking
        """
        self.model_path = model_path
        self.observe_only = observe_only
        self.model = None
        
        # Statistics
        self.packets_seen = 0
        self.packets_allowed = 0
        self.packets_blocked = 0
        self.blocked_ips = set()
        
        self.start_time = time.time()
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load trained RL model."""
        try:
            self.model = DQN.load(self.model_path)
            print(f"[SUCCESS] Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"[ERROR] Model not found at {self.model_path}")
            print("[INFO] Please train a model first:")
            print("       python -m firewall.train")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
    
    def handle_packet(self, packet):
        """
        Process packet and make AI decision.
        
        Args:
            packet: Scapy packet object
        """
        # Only process IP packets
        if not packet.haslayer(IP):
            return
        
        self.packets_seen += 1
        
        # Extract source IP
        ip_src = packet[IP].src
        
        # Extract features
        features = extract_features_from_packet(packet)
        
        # Get AI decision
        action, _states = self.model.predict(features, deterministic=True)
        
        # action: 0 = Allow, 1 = Block
        decision = "BLOCK" if action == 1 else "ALLOW"
        
        # Update statistics
        if action == 0:
            self.packets_allowed += 1
        else:
            self.packets_blocked += 1
        
        # Display decision
        protocol = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "OTHER"
        
        color_code = "\033[91m" if action == 1 else "\033[92m"  # Red for block, green for allow
        reset_code = "\033[0m"
        
        print(f"\n[Packet #{self.packets_seen}] {color_code}[{decision}]{reset_code}")
        print(f"  Source IP: {ip_src}")
        print(f"  Protocol:  {protocol}")
        print(f"  Features:  {features}")
        
        # Execute action
        if action == 1 and not self.observe_only:
            if ip_src not in self.blocked_ips:
                block_ip(ip_src)
                self.blocked_ips.add(ip_src)
                print(f"  ⚠️  IP {ip_src} has been BLOCKED via iptables")
        
        if self.observe_only and action == 1:
            print(f"  ℹ️  Would block {ip_src} (OBSERVE mode)")
    
    def start(self, packet_count=0, interface=None):
        """
        Start adaptive firewall.
        
        Args:
            packet_count: Number of packets to process (0 = unlimited)
            interface: Network interface to monitor
        """
        mode = "OBSERVE ONLY" if self.observe_only else "ACTIVE BLOCKING"
        
        print("="*60)
        print("ADAPTIVE FIREWALL AI - POLICY ENFORCEMENT")
        print("="*60)
        print(f"\n[INFO] Mode: {mode}")
        print(f"[INFO] Model: {self.model_path}")
        print(f"[INFO] Interface: {interface or 'default'}")
        print(f"[INFO] Press Ctrl+C to stop\n")
        
        if not self.observe_only:
            print("⚠️  WARNING: Active blocking enabled!")
            print("⚠️  The firewall will block IPs flagged as malicious")
            print("⚠️  Press Ctrl+C within 5 seconds to cancel...\n")
            time.sleep(5)
        
        try:
            sniff(
                prn=self.handle_packet,
                count=packet_count,
                iface=interface,
                store=False
            )
        except KeyboardInterrupt:
            print("\n\n[INFO] Firewall stopped by user")
        except PermissionError:
            print("\n[ERROR] Permission denied!")
            print("[INFO] Run with administrator privileges:")
            print("       sudo python -m runtime.policy")
            return
        except Exception as e:
            print(f"\n[ERROR] Firewall error: {e}")
            return
        
        self._print_summary()
    
    def _print_summary(self):
        """Print execution summary."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("FIREWALL STATISTICS")
        print("="*60)
        print(f"Total packets:    {self.packets_seen}")
        print(f"Allowed:          {self.packets_allowed} ({self.packets_allowed/max(self.packets_seen,1)*100:.1f}%)")
        print(f"Blocked:          {self.packets_blocked} ({self.packets_blocked/max(self.packets_seen,1)*100:.1f}%)")
        print(f"Unique IPs blocked: {len(self.blocked_ips)}")
        print(f"Time elapsed:     {elapsed:.2f} seconds")
        print(f"Rate:             {self.packets_seen/elapsed:.2f} packets/sec")
        print("="*60 + "\n")
        
        if self.blocked_ips and not self.observe_only:
            print("Blocked IPs:", ", ".join(self.blocked_ips))
            print("\n[INFO] To unblock all IPs, run:")
            print("       python -m runtime.firewall_controller --unblock-all")


def main():
    """Main entry point."""
    
    # Configuration
    OBSERVE_ONLY = True  # ⚠️ SET TO False TO ENABLE REAL BLOCKING
    PACKET_COUNT = 100   # Number of packets to process (0 = unlimited)
    
    print("\n⚠️  SAFETY NOTICE")
    print("="*60)
    print(f"Running in: {'OBSERVE ONLY mode' if OBSERVE_ONLY else '🔴 ACTIVE BLOCKING mode'}")
    
    if OBSERVE_ONLY:
        print("[SAFE] No actual blocking will occur")
        print("[INFO] To enable blocking, edit OBSERVE_ONLY in runtime/policy.py")
    else:
        print("⚠️  Real iptables rules will be created!")
        print("⚠️  This may disrupt network connectivity!")
    
    print("="*60 + "\n")
    
    # Create firewall instance
    firewall = AdaptiveFirewall(
        model_path="model/firewall_dqn",
        observe_only=OBSERVE_ONLY
    )
    
    # Start processing
    firewall.start(packet_count=PACKET_COUNT)


if __name__ == "__main__":
    main()
