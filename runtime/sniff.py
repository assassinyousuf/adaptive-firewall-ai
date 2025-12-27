"""
Live packet capture module using Scapy.

STAGE 2: Observation mode - captures and displays packets without blocking.

Usage:
    sudo python -m runtime.sniff
    
Note: Requires administrator/root privileges for packet capture.
"""

from scapy.all import sniff, IP, TCP, UDP, ICMP
from firewall.features import extract_features_from_packet
import time


class PacketObserver:
    """Observes network packets and extracts features."""
    
    def __init__(self, interface=None, packet_count=100):
        """
        Initialize packet observer.
        
        Args:
            interface: Network interface to sniff on (None = default)
            packet_count: Number of packets to capture (0 = infinite)
        """
        self.interface = interface
        self.packet_count = packet_count
        self.packets_seen = 0
        self.start_time = time.time()
        
    def handle_packet(self, packet):
        """
        Process captured packet.
        
        Args:
            packet: Scapy packet object
        """
        self.packets_seen += 1
        
        # Check if packet has IP layer
        if not packet.haslayer(IP):
            return
        
        # Extract IP information
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        
        # Extract features
        features = extract_features_from_packet(packet)
        
        # Determine protocol name
        protocol = "OTHER"
        if packet.haslayer(TCP):
            protocol = "TCP"
        elif packet.haslayer(UDP):
            protocol = "UDP"
        elif packet.haslayer(ICMP):
            protocol = "ICMP"
        
        # Display packet information
        print(f"\n[Packet #{self.packets_seen}]")
        print(f"  Source:      {ip_src}")
        print(f"  Destination: {ip_dst}")
        print(f"  Protocol:    {protocol}")
        print(f"  Size:        {features[0]:.0f} bytes")
        print(f"  Features:    {features}")
        
    def start_sniffing(self):
        """Start capturing packets."""
        print("="*60)
        print("ADAPTIVE FIREWALL AI - PACKET OBSERVER")
        print("="*60)
        print(f"\n[INFO] Starting packet capture...")
        print(f"[INFO] Interface: {self.interface or 'default'}")
        print(f"[INFO] Count: {self.packet_count if self.packet_count > 0 else 'unlimited'}")
        print(f"[INFO] Press Ctrl+C to stop\n")
        
        try:
            sniff(
                prn=self.handle_packet,
                count=self.packet_count,
                iface=self.interface,
                store=False
            )
        except KeyboardInterrupt:
            print("\n\n[INFO] Capture interrupted by user")
        except PermissionError:
            print("\n[ERROR] Permission denied!")
            print("[INFO] Run with administrator privileges:")
            print("       sudo python -m runtime.sniff")
            return
        except Exception as e:
            print(f"\n[ERROR] Capture failed: {e}")
            return
        
        # Summary
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("CAPTURE SUMMARY")
        print("="*60)
        print(f"Packets captured: {self.packets_seen}")
        print(f"Time elapsed:     {elapsed:.2f} seconds")
        print(f"Rate:             {self.packets_seen/elapsed:.2f} packets/sec")
        print("="*60 + "\n")


def main():
    """Main entry point for packet observer."""
    
    # Create observer
    observer = PacketObserver(packet_count=50)  # Capture 50 packets for demo
    
    # Start sniffing
    observer.start_sniffing()
    
    print("[INFO] To capture more packets, edit packet_count in runtime/sniff.py")
    print("[INFO] Next step: Train the model with: python -m firewall.train")


if __name__ == "__main__":
    main()
