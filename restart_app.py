#!/usr/bin/env python3
"""
Simple script to restart the Streamlit app
"""
import os
import sys
import subprocess
import signal
import time

def restart_streamlit():
    """Kill existing Streamlit processes and restart"""
    print("🔄 Restarting Streamlit app...")
    
    # Kill existing streamlit processes
    try:
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        print("✅ Killed existing Streamlit processes")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Could not kill processes: {e}")
    
    # Start new streamlit process
    try:
        print("🚀 Starting new Streamlit app...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "media_planner_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    restart_streamlit() 