import subprocess
import sys
import time
import os
import socket
import webbrowser
import logging
import platform
from pathlib import Path

# ------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------
LOG_FILE = Path(__file__).parent / "orchestrator.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------------------------------------------
# Detect environment
# ------------------------------------------------------------
IS_WINDOWS = platform.system().lower().startswith("win")

# ------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
KAFKA_BAT = PROJECT_ROOT / "scripts" / "start-kafka.bat"
PROM_BAT = PROJECT_ROOT / "scripts" / "start-prometheus.bat"
PRODUCER = PROJECT_ROOT / "streaming" / "synthetic-producer.py"
CONSUMER = PROJECT_ROOT / "streaming" / "kafka_consumer.py"

if IS_WINDOWS:
    BROKER_HOST, PROM_HOST = "localhost", "localhost"
    GRAFANA_URL = (
        "http://localhost:3000/d/0b1ea385-cc05-4c5c-98a2-0bb6e170b35b/"
        "nids-dashboard-vigilix-kafka-monitoring?orgId=1&from=now-5m&to=now"
        "&timezone=browser&refresh=10s"
    )
else:
    # Inside Docker or Linux
    BROKER_HOST, PROM_HOST = "kafka", "prometheus"
    GRAFANA_URL = (
        "http://grafana:3000/d/0b1ea385-cc05-4c5c-98a2-0bb6e170b35b/"
        "nids-dashboard-vigilix-kafka-monitoring?orgId=1&from=now-5m&to=now"
        "&timezone=browser&refresh=10s"
    )

BROKER_PORT, PROM_PORT = 9092, 9090

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def wait_for_port(host, port, timeout=90):
    """Waits for the specified port to be open on the given host within the timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def start_bat(path: Path, title: str):
    """Starts a .bat file in a new terminal window."""
    if not path.exists():
        logging.error(f"Error: Missing {path}")
        sys.exit(1)

    logging.info(f"Starting: {path.name}")
    try:
        bat_command = f'cd /d "{path.parent}" && start "{title}" "{path.name}"'
        os.system(bat_command)
        time.sleep(3)
    except Exception as e:
        logging.error(f"Error starting {title}: {e}")
        sys.exit(1)


def start_py(path: Path, title: str, args=None):
    """Starts a Python script in a new terminal window."""
    if not path.exists():
        logging.error(f"Error: Missing {path}")
        sys.exit(1)

    args = args or []
    logging.info(f"Starting: {path.name}")
    try:
        if IS_WINDOWS:
            args_str = " ".join([f'"{arg}"' for arg in args])
            py_command = f'cd /d "{path.parent}" && start "{title}" "{sys.executable}" "{path.name}" {args_str}'
            os.system(py_command)
        else:
            # On Linux (e.g. Docker), run directly in background
            subprocess.Popen([sys.executable, str(path)] + args)
        time.sleep(2)
    except Exception as e:
        logging.error(f"Error starting {title}: {e}")
        sys.exit(1)


def check_prerequisites():
    """Check if all required components are available."""
    logging.info("Checking prerequisites...")

    # Python check
    try:
        subprocess.run([sys.executable, "--version"], capture_output=True, check=True)
    except Exception:
        logging.error("Error: Python not found or not accessible")
        return False

    # File existence check
    required_files = [PRODUCER, CONSUMER]
    if IS_WINDOWS:
        required_files += [KAFKA_BAT, PROM_BAT]

    missing_files = [str(f) for f in required_files if not f.exists()]
    if missing_files:
        logging.error("Error: Missing required files:")
        for f in missing_files:
            logging.error(f"  - {f}")
        return False

    logging.info("All prerequisites satisfied ✓")
    return True


# ------------------------------------------------------------
# Main Orchestration
# ------------------------------------------------------------
def main():
    logging.info("=" * 60)
    logging.info("VIGILIX REAL-TIME ANOMALY DETECTION PIPELINE")
    logging.info("=" * 60)

    # 🌐 Log current environment
    if IS_WINDOWS:
        logging.info("Environment detected: 🪟 Running in WINDOWS mode")
        print("🪟 Running in WINDOWS mode — using .bat scripts for Kafka & Prometheus")
    else:
        logging.info("Environment detected: 🐳 Running in DOCKER / LINUX mode")
        print("🐳 Running in DOCKER / LINUX mode — using Docker containers for Kafka & Prometheus")

    if not check_prerequisites():
        logging.error("Please ensure all components are properly set up.")
        sys.exit(1)

    logging.info("\nStarting components...")

    # Kafka startup
    logging.info("\n1. Starting Zookeeper/Kafka...")
    if IS_WINDOWS:
        start_bat(KAFKA_BAT, "Vigilix-Kafka")
    else:
        logging.info("Kafka is managed by Docker Compose. Skipping manual startup.")

    logging.info("\n2. Waiting for Kafka to be ready...")
    if wait_for_port(BROKER_HOST, BROKER_PORT, timeout=120):
        logging.info("✓ Kafka is ready.")
    else:
        logging.warning("⚠ Kafka not detected on 9092. Producer/consumer may fail.")

    # Prometheus startup
    logging.info("\n3. Starting Prometheus...")
    if IS_WINDOWS:
        start_bat(PROM_BAT, "Vigilix-Prometheus")
    else:
        logging.info("Prometheus is managed by Docker Compose. Skipping manual startup.")

    logging.info("\n4. Waiting for Prometheus to be ready...")
    if wait_for_port(PROM_HOST, PROM_PORT, timeout=60):
        logging.info("✓ Prometheus is ready.")
    else:
        logging.warning("⚠ Prometheus not detected on 9090.")

    # Start producer and consumer
    logging.info("\n5. Starting synthetic data producer...")
    start_py(PRODUCER, "Vigilix-Producer")

    logging.info("\n6. Starting Kafka consumer (metrics on :8001)...")
    start_py(CONSUMER, "Vigilix-Consumer")

    # Open Grafana dashboard (only on Windows)
    if IS_WINDOWS:
        logging.info(f"\n7. Opening Grafana at {GRAFANA_URL}...")
        webbrowser.open(GRAFANA_URL)
    else:
        logging.info(f"\nGrafana available at {GRAFANA_URL}")

    logging.info("\n" + "=" * 60)
    logging.info("VIGILIX PIPELINE STARTED SUCCESSFULLY!")
    logging.info("=" * 60)
    logging.info("\nComponents running:")
    logging.info(f"- Kafka Host: {BROKER_HOST}:{BROKER_PORT}")
    logging.info(f"- Prometheus Host: {PROM_HOST}:{PROM_PORT}")
    logging.info(f"- Grafana URL: {GRAFANA_URL}")
    logging.info("- Producer and Consumer running")

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("Shutting down all components...")
        stop_all()


# ------------------------------------------------------------
# Graceful shutdown
# ------------------------------------------------------------
def stop_all():
    """Attempts to gracefully stop all components."""
    logging.info("\nStopping all Vigilix components...")
    if IS_WINDOWS:
        try:
            subprocess.run(["taskkill", "/f", "/im", "java.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "prometheus.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "cmd.exe"], capture_output=True)
        except Exception as e:
            logging.error(f"Error stopping processes: {e}")
    else:
        logging.info("On Docker/Linux, components are managed externally (via Docker Compose).")

    logging.info("All components stopped. Goodbye!")


if __name__ == "__main__":
    main()