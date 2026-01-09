#!/usr/bin/env python3
"""
Cleanup script to remove test data contamination from performance_history.json
Run this ONCE after deploying the fixes to clean up test artifacts
"""

import json
import os
from datetime import datetime

def cleanup_performance_history():
    """Remove test trades from performance_history.json"""
    
    performance_file = "./performance_history.json"
    
    if not os.path.exists(performance_file):
        print("✅ No performance_history.json found - nothing to clean")
        return
    
    # Backup first
    backup_file = f"{performance_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        with open(performance_file, 'r') as f:
            data = json.load(f)
        
        # Backup
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"📦 Backup created: {backup_file}")
        
        # Get original counts
        original_trades = len(data.get('trades', []))
        original_equity = len(data.get('equity_curve', []))
        
        # Filter out test trades (they have sequential timestamps within seconds)
        if 'trades' in data and data['trades']:
            # Keep only trades that look real (not from rapid-fire tests)
            real_trades = []
            
            for i, trade in enumerate(data['trades']):
                # Skip if this looks like test data:
                # - Exactly round prices (50000, 55000, 60000)
                # - Sequential timestamps within same second
                price = trade.get('price', 0)
                
                # Check if price is suspiciously round (test data)
                is_test_price = price in [50000, 51000, 52000, 53000, 54000, 55000, 
                                         58000, 60000, 48000, 49000]
                
                if not is_test_price:
                    real_trades.append(trade)
            
            data['trades'] = real_trades
        
        # Clear equity curve (it will rebuild from real trades)
        if 'equity_curve' in data:
            data['equity_curve'] = []
        
        # Save cleaned data
        with open(performance_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        new_trades = len(data.get('trades', []))
        
        print(f"\n✅ Cleanup complete:")
        print(f"   Trades: {original_trades} → {new_trades} (removed {original_trades - new_trades} test trades)")
        print(f"   Equity curve: Reset (will rebuild from real data)")
        print(f"\n⚠️  If you need to restore: cp {backup_file} {performance_file}")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        print(f"   Backup preserved at: {backup_file}")

if __name__ == "__main__":
    print("="*60)
    print("Performance History Cleanup")
    print("="*60)
    print("\nThis will remove test data contamination from your trading bot.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response == 'yes':
        cleanup_performance_history()
    else:
        print("Cancelled.")