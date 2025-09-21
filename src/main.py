import subprocess
import sys
import time
import os
import socket
import webbrowser
from pathlib import Path

# Get the project root directory (main.py is in src/, so go up one level)
PROJECT_ROOT = Path(__file__).parent.parent

# Absolute paths for Kafka and Prometheus
KAFKA_BAT = PROJECT_ROOT / "scripts" / "start-kafka.bat"
PROM_BAT = PROJECT_ROOT / "scripts" / "start-prometheus.bat"
PRODUCER = PROJECT_ROOT / "streaming" / "synthetic-producer.py"
CONSUMER = PROJECT_ROOT / "streaming" / "kafka_consumer.py"

BROKER_HOST, BROKER_PORT = "localhost", 9092
PROM_HOST, PROM_PORT = "localhost", 9090
GRAFANA_URL = "http://localhost:3000"

def wait_for_port(host, port, timeout=90):
    """
    Waits for the specified port to be open on the given host within the timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False

def start_bat(path: Path, title: str):
    """
    Starts a .bat file in a new terminal window.
    """
    if not path.exists():
        print(f"Error: Missing {path}")
        print(f"Please ensure the file exists at: {path.absolute()}")
        sys.exit(1)
    
    print(f"Starting: {path.name}")
    try:
        # Use os.system with proper quoting to handle special characters
        bat_command = f'cd /d "{path.parent}" && start "{title}" "{path.name}"'
        os.system(bat_command)
        time.sleep(3)  # Give it a moment to start
    except Exception as e:
        print(f"Error starting {title}: {e}")
        sys.exit(1)

def start_py(path: Path, title: str, args=None):
    """
    Starts a Python script in a new terminal window.
    """
    if not path.exists():
        print(f"Error: Missing {path}")
        print(f"Please ensure the file exists at: {path.absolute()}")
        sys.exit(1)
    
    args = args or []
    print(f"Starting: {path.name}")
    try:
        # Build the command properly with proper quoting
        args_str = " ".join([f'"{arg}"' for arg in args])
        py_command = f'cd /d "{path.parent}" && start "{title}" "{sys.executable}" "{path.name}" {args_str}'
        os.system(py_command)
        time.sleep(2)  # Give it a moment to start
    except Exception as e:
        print(f"Error starting {title}: {e}")
        sys.exit(1)

def check_prerequisites():
    """
    Check if all required components are available
    """
    print("Checking prerequisites...")
    
    # Check if Python is available
    try:
        subprocess.run([sys.executable, "--version"], capture_output=True, check=True)
    except:
        print("Error: Python not found or not accessible")
        return False
    
    # Check if required files exist
    required_files = [KAFKA_BAT, PROM_BAT, PRODUCER, CONSUMER]
    missing_files = [str(f) for f in required_files if not f.exists()]
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("All prerequisites satisfied ✓")
    return True

def main():
    print("=" * 60)
    print("VIGILIX REAL-TIME ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    
    # Check prerequisites first
    if not check_prerequisites():
        print("\nPlease ensure all components are properly set up.")
        print("Refer to README.md for setup instructions.")
        sys.exit(1)
    
    print("\nStarting components...")

    # Start Kafka and Zookeeper
    print("\n1. Starting Zookeeper/Kafka...")
    start_bat(KAFKA_BAT, "Vigilix-Kafka")

    # Wait for Kafka to be ready
    print("\n2. Waiting for Kafka on localhost:9092 ...")
    if wait_for_port(BROKER_HOST, BROKER_PORT, timeout=120):
        print("✓ Kafka is ready.")
    else:
        print("⚠ Warning: Kafka not detected on 9092. Producer/consumer may fail.")
        print("Continuing anyway...")

    # Start Prometheus
    print("\n3. Starting Prometheus...")
    start_bat(PROM_BAT, "Vigilix-Prometheus")

    # Wait for Prometheus to be ready
    print("\n4. Waiting for Prometheus on localhost:9090 ...")
    if wait_for_port(PROM_HOST, PROM_PORT, timeout=60):
        print("✓ Prometheus is ready.")
    else:
        print("⚠ Warning: Prometheus not detected on 9090.")
        print("Continuing anyway...")

    # Start synthetic producer
    print("\n5. Starting synthetic data producer...")
    start_py(PRODUCER, "Vigilix-Producer")

    # Start Kafka consumer
    print("\n6. Starting Kafka consumer (metrics on :8001)...")
    start_py(CONSUMER, "Vigilix-Consumer")

    # Open Grafana dashboard
    print(f"\n7. Opening Grafana at {GRAFANA_URL}...")
    webbrowser.open(GRAFANA_URL)

    # Keep the orchestrator running to prevent closing
    print("\n" + "=" * 60)
    print("VIGILIX PIPELINE STARTED SUCCESSFULLY!")
    print("=" * 60)
    print("\nComponents running:")
    print("- Kafka & Zookeeper: scripts/start-kafka.bat window")
    print("- Prometheus: scripts/start-prometheus.bat window")
    print("- Producer: streaming/synthetic-producer.py window")
    print("- Consumer: streaming/kafka_consumer.py window")
    print(f"- Grafana Dashboard: {GRAFANA_URL}")
    
    print("\nPress Ctrl+C to stop all components and exit")
    print("-" * 40)
    
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting orchestrator. Stopping components...")
        stop_all()

def stop_all():
    """
    Attempts to gracefully stop all components
    """
    print("\nStopping all Vigilix components...")
    
    try:
        # This is a basic approach - you might want to implement more sophisticated
        # process management if needed
        subprocess.run(["taskkill", "/f", "/im", "java.exe"], capture_output=True)
        subprocess.run(["taskkill", "/f", "/im", "prometheus.exe"], capture_output=True)
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
        subprocess.run(["taskkill", "/f", "/im", "cmd.exe"], capture_output=True)
    except Exception as e:
        print(f"Error stopping processes: {e}")
    
    print("All components stopped. Goodbye!")

if __name__ == "__main__":
    main()