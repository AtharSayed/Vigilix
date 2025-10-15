import subprocess
import sys
import time
import os
import socket
import webbrowser
import logging
from pathlib import Path

# Set up logging
LOG_FILE = Path(__file__).parent / "orchestrator.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # Set the logging level to INFO to log detailed messages
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Get the project root directory (main.py is in src/, so go up one level)
PROJECT_ROOT = Path(__file__).parent.parent

# Absolute paths for Kafka and Prometheus
KAFKA_BAT = PROJECT_ROOT / "scripts" / "start-kafka.bat"
PROM_BAT = PROJECT_ROOT / "scripts" / "start-prometheus.bat"
PRODUCER = PROJECT_ROOT / "streaming" / "synthetic-producer.py"
CONSUMER = PROJECT_ROOT / "streaming" / "kafka_consumer.py"

BROKER_HOST, BROKER_PORT = "localhost", 9092
PROM_HOST, PROM_PORT = "localhost", 9090
GRAFANA_URL = "http://localhost:3000/d/0b1ea385-cc05-4c5c-98a2-0bb6e170b35b/nids-dashboard-vigilix-kafka-monitoring?orgId=1&from=now-5m&to=now&timezone=browser&refresh=10s"

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
        logging.error(f"Error: Missing {path}")
        logging.error(f"Please ensure the file exists at: {path.absolute()}")
        sys.exit(1)
    
    logging.info(f"Starting: {path.name}")
    try:
        # Use os.system with proper quoting to handle special characters
        bat_command = f'cd /d "{path.parent}" && start "{title}" "{path.name}"'
        os.system(bat_command)
        time.sleep(3)  # Give it a moment to start
    except Exception as e:
        logging.error(f"Error starting {title}: {e}")
        sys.exit(1)

def start_py(path: Path, title: str, args=None):
    if not path.exists():
        logging.error(f"Error: Missing {path}")
        logging.error(f"Please ensure the file exists at: {path.absolute()}")
        sys.exit(1)
    
    args = args or []
    logging.info(f"Starting: {path.name}")
    try:
        # Build the command properly with proper quoting
        args_str = " ".join([f'"{arg}"' for arg in args])
        py_command = f'cd /d "{path.parent}" && start "{title}" "{sys.executable}" "{path.name}" {args_str}'
        os.system(py_command)
        time.sleep(2)  # Give it a moment to start
    except Exception as e:
        logging.error(f"Error starting {title}: {e}")
        sys.exit(1)

def check_prerequisites():
    """
    Check if all required components are available
    """
    logging.info("Checking prerequisites...")
    
    # Checking if Python is available
    try:
        subprocess.run([sys.executable, "--version"], capture_output=True, check=True)
    except:
        logging.error("Error: Python not found or not accessible")
        return False
    
    # Check if required files exist
    required_files = [KAFKA_BAT, PROM_BAT, PRODUCER, CONSUMER]
    missing_files = [str(f) for f in required_files if not f.exists()]
    
    if missing_files:
        logging.error("Error: Missing required files:")
        for file in missing_files:
            logging.error(f"  - {file}")
        return False
    
    logging.info("All prerequisites satisfied ✓")
    return True

def main():
    logging.info("=" * 60)
    logging.info("VIGILIX REAL-TIME ANOMALY DETECTION PIPELINE")
    logging.info("=" * 60)
    
    # Check prerequisites first
    if not check_prerequisites():
        logging.error("\nPlease ensure all components are properly set up.")
        logging.error("Refer to README.md for setup instructions.")
        sys.exit(1)
    
    logging.info("\nStarting components...")

    # Start Kafka and Zookeeper
    logging.info("\n1. Starting Zookeeper/Kafka...")
    start_bat(KAFKA_BAT, "Vigilix-Kafka")

    # Wait for Kafka to be ready
    logging.info("\n2. Waiting for Kafka on localhost:9092 ...")
    if wait_for_port(BROKER_HOST, BROKER_PORT, timeout=120):
        logging.info("✓ Kafka is ready.")
    else:
        logging.warning("⚠ Warning: Kafka not detected on 9092. Producer/consumer may fail.")
        logging.warning("Continuing anyway...")

    # Start Prometheus
    logging.info("\n3. Starting Prometheus...")
    start_bat(PROM_BAT, "Vigilix-Prometheus")

    # Wait for Prometheus to be ready
    logging.info("\n4. Waiting for Prometheus on localhost:9090 ...")
    if wait_for_port(PROM_HOST, PROM_PORT, timeout=60):
        logging.info("✓ Prometheus is ready.")
    else:
        logging.warning("⚠ Warning: Prometheus not detected on 9090.")
        logging.warning("Continuing anyway...")

    # Start synthetic producer
    logging.info("\n5. Starting synthetic data producer...")
    start_py(PRODUCER, "Vigilix-Producer")

    # Start Kafka consumer
    logging.info("\n6. Starting Kafka consumer (metrics on :8001)...")
    start_py(CONSUMER, "Vigilix-Consumer")

    # Open Grafana dashboard
    logging.info(f"\n7. Opening Grafana at {GRAFANA_URL}...")
    webbrowser.open(GRAFANA_URL)

    # Keep the orchestrator running to prevent closing
    logging.info("\n" + "=" * 60)
    logging.info("VIGILIX PIPELINE STARTED SUCCESSFULLY!")
    logging.info("=" * 60)
    logging.info("\nComponents running:")
    logging.info("- Kafka & Zookeeper: scripts/start-kafka.bat window")
    logging.info("- Prometheus: scripts/start-prometheus.bat window")
    logging.info("- Producer: streaming/synthetic-producer.py window")
    logging.info("- Consumer: streaming/kafka_consumer.py window")
    logging.info(f"- Grafana Dashboard: {GRAFANA_URL}")
    
    logging.info("\nPress Ctrl+C to stop all components and exit")
    logging.info("-" * 40)
    
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("\nExiting orchestrator. Stopping components...")
        stop_all()

def stop_all():
    """
    Attempts to gracefully stop all components
    """
    logging.info("\nStopping all Vigilix components...")
    
    try:
        subprocess.run(["taskkill", "/f", "/im", "java.exe"], capture_output=True)
        subprocess.run(["taskkill", "/f", "/im", "prometheus.exe"], capture_output=True)
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
        subprocess.run(["taskkill", "/f", "/im", "cmd.exe"], capture_output=True)
    except Exception as e:
        logging.error(f"Error stopping processes: {e}")
    
    logging.info("All components stopped. Goodbye!")

if __name__ == "__main__":
    main()