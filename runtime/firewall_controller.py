"""
System firewall controller using iptables (Linux) or Windows Firewall.

Handles low-level firewall rule management.

⚠️ WARNING: This module modifies system firewall rules.
   Use with caution and proper authorization.
"""

import subprocess
import platform
import sys


class FirewallController:
    """Cross-platform firewall controller."""
    
    def __init__(self):
        self.system = platform.system()
        self.blocked_ips = set()
        
    def block_ip(self, ip_address):
        """
        Block an IP address.
        
        Args:
            ip_address: IP address to block
            
        Returns:
            True if successful, False otherwise
        """
        if self.system == "Linux":
            return self._block_ip_linux(ip_address)
        elif self.system == "Windows":
            return self._block_ip_windows(ip_address)
        else:
            print(f"[ERROR] Unsupported operating system: {self.system}")
            return False
    
    def _block_ip_linux(self, ip_address):
        """Block IP using iptables (Linux)."""
        try:
            cmd = f"iptables -A INPUT -s {ip_address} -j DROP"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.blocked_ips.add(ip_address)
                print(f"[FIREWALL] Blocked {ip_address} via iptables")
                return True
            else:
                print(f"[ERROR] Failed to block {ip_address}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Exception blocking {ip_address}: {e}")
            return False
    
    def _block_ip_windows(self, ip_address):
        """Block IP using Windows Firewall."""
        try:
            rule_name = f"AdaptiveFirewall_Block_{ip_address.replace('.', '_')}"
            
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule_name}",
                "dir=in",
                "action=block",
                f"remoteip={ip_address}"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.blocked_ips.add(ip_address)
                print(f"[FIREWALL] Blocked {ip_address} via Windows Firewall")
                return True
            else:
                print(f"[ERROR] Failed to block {ip_address}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Exception blocking {ip_address}: {e}")
            return False
    
    def unblock_ip(self, ip_address):
        """
        Unblock an IP address.
        
        Args:
            ip_address: IP address to unblock
            
        Returns:
            True if successful, False otherwise
        """
        if self.system == "Linux":
            return self._unblock_ip_linux(ip_address)
        elif self.system == "Windows":
            return self._unblock_ip_windows(ip_address)
        else:
            return False
    
    def _unblock_ip_linux(self, ip_address):
        """Unblock IP using iptables (Linux)."""
        try:
            cmd = f"iptables -D INPUT -s {ip_address} -j DROP"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.blocked_ips.discard(ip_address)
                print(f"[FIREWALL] Unblocked {ip_address}")
                return True
            else:
                print(f"[ERROR] Failed to unblock {ip_address}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Exception unblocking {ip_address}: {e}")
            return False
    
    def _unblock_ip_windows(self, ip_address):
        """Unblock IP using Windows Firewall."""
        try:
            rule_name = f"AdaptiveFirewall_Block_{ip_address.replace('.', '_')}"
            
            cmd = [
                "netsh", "advfirewall", "firewall", "delete", "rule",
                f"name={rule_name}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.blocked_ips.discard(ip_address)
                print(f"[FIREWALL] Unblocked {ip_address}")
                return True
            else:
                print(f"[ERROR] Failed to unblock {ip_address}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Exception unblocking {ip_address}: {e}")
            return False
    
    def unblock_all(self):
        """Unblock all IPs that were blocked by this controller."""
        print(f"\n[INFO] Unblocking {len(self.blocked_ips)} IP addresses...")
        
        for ip in list(self.blocked_ips):
            self.unblock_ip(ip)
        
        print("[SUCCESS] All IPs unblocked\n")
    
    def list_blocked_ips(self):
        """Return list of blocked IPs."""
        return list(self.blocked_ips)


# Global controller instance
_controller = FirewallController()


def block_ip(ip_address):
    """Block an IP address using the global controller."""
    return _controller.block_ip(ip_address)


def unblock_ip(ip_address):
    """Unblock an IP address using the global controller."""
    return _controller.unblock_ip(ip_address)


def unblock_all():
    """Unblock all IPs."""
    return _controller.unblock_all()


def get_blocked_ips():
    """Get list of blocked IPs."""
    return _controller.list_blocked_ips()


def main():
    """Command-line interface for firewall controller."""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m runtime.firewall_controller --block <IP>")
        print("  python -m runtime.firewall_controller --unblock <IP>")
        print("  python -m runtime.firewall_controller --unblock-all")
        print("  python -m runtime.firewall_controller --list")
        return
    
    command = sys.argv[1]
    
    if command == "--block" and len(sys.argv) == 3:
        ip = sys.argv[2]
        print(f"[INFO] Blocking {ip}...")
        block_ip(ip)
        
    elif command == "--unblock" and len(sys.argv) == 3:
        ip = sys.argv[2]
        print(f"[INFO] Unblocking {ip}...")
        unblock_ip(ip)
        
    elif command == "--unblock-all":
        unblock_all()
        
    elif command == "--list":
        blocked = get_blocked_ips()
        print(f"\nCurrently blocked IPs ({len(blocked)}):")
        for ip in blocked:
            print(f"  - {ip}")
        print()
        
    else:
        print("[ERROR] Invalid command")


if __name__ == "__main__":
    main()
