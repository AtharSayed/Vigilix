import pandas as pd
import numpy as np
import random
import time
from datetime import datetime
import requests

def generate_realistic_network_traffic(num_samples=10, anomaly_ratio=0.2):
    """
    Generate more realistic network traffic data based on CIC dataset patterns
    """
    # Load original dataset to get realistic value ranges
    try:
        sample_df = pd.read_parquet(r"C:\Users\sayed\Desktop\L&T-Project\Vigilix\data\raw\cic-collection.parquet\cic-collection.parquet", engine="pyarrow")
        feature_columns = [col for col in sample_df.columns if col not in ['Label', 'ClassLabel']]
        
        # Get statistical properties for more realistic data generation
        stats_df = sample_df.describe()
    except:
        # Fallback if we can't load the dataset
        feature_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
            'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
            'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
            'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
            'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
            'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
            'Bwd Packets/s', 'Packet Length Max', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'SYN Flag Count',
            'URG Flag Count', 'Avg Packet Size', 'Avg Fwd Segment Size',
            'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes',
            'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
    
    data = []
    
    for i in range(num_samples):
        is_anomaly = random.random() < anomaly_ratio
        
        sample = {}
        
        # Generate realistic baseline values
        if not is_anomaly:
            # Normal traffic patterns
            sample['Flow Duration'] = random.randint(1000, 10000)
            sample['Total Fwd Packets'] = random.randint(5, 50)
            sample['Total Backward Packets'] = random.randint(3, 40)
            sample['Fwd Packets Length Total'] = random.randint(500, 5000)
            sample['Bwd Packets Length Total'] = random.randint(300, 4000)
            sample['Fwd Packet Length Max'] = random.randint(100, 1500)
            sample['Fwd Packet Length Mean'] = random.uniform(50, 800)
            sample['Fwd Packet Length Std'] = random.uniform(10, 100)
            sample['Bwd Packet Length Max'] = random.randint(100, 1400)
            sample['Bwd Packet Length Mean'] = random.uniform(40, 750)
            sample['Bwd Packet Length Std'] = random.uniform(8, 90)
            sample['Flow Bytes/s'] = random.randint(5000, 50000)
            sample['Flow Packets/s'] = random.randint(10, 100)
            sample['Flow IAT Mean'] = random.uniform(0.01, 0.1)
            sample['Flow IAT Std'] = random.uniform(0.005, 0.05)
            sample['Flow IAT Max'] = random.uniform(0.05, 0.3)
            sample['Flow IAT Min'] = random.uniform(0.001, 0.02)
        else:
            # Anomalous traffic patterns (DDoS/attack patterns)
            sample['Flow Duration'] = random.randint(100, 1000)  # Shorter duration
            sample['Total Fwd Packets'] = random.randint(100, 1000)  # Many more packets
            sample['Total Backward Packets'] = random.randint(1, 10)  # Asymmetric
            sample['Fwd Packets Length Total'] = random.randint(10000, 50000)  # Large data transfer
            sample['Bwd Packets Length Total'] = random.randint(100, 1000)  # Small responses
            sample['Fwd Packet Length Max'] = random.randint(1400, 1500)  # Max size packets
            sample['Fwd Packet Length Mean'] = random.uniform(1000, 1450)
            sample['Fwd Packet Length Std'] = random.uniform(1, 10)  # Very consistent
            sample['Bwd Packet Length Max'] = random.randint(40, 100)  # Small packets
            sample['Bwd Packet Length Mean'] = random.uniform(30, 80)
            sample['Bwd Packet Length Std'] = random.uniform(1, 5)
            sample['Flow Bytes/s'] = random.randint(100000, 1000000)  # High bandwidth
            sample['Flow Packets/s'] = random.randint(500, 5000)  # High packet rate
            sample['Flow IAT Mean'] = random.uniform(0.0001, 0.001)  # Very frequent
            sample['Flow IAT Std'] = random.uniform(0.0001, 0.0005)
            sample['Flow IAT Max'] = random.uniform(0.001, 0.005)
            sample['Flow IAT Min'] = random.uniform(0.00001, 0.0001)
        
        # Fill in remaining features with reasonable values
        sample['Fwd IAT Total'] = sample['Flow Duration'] / 1000  # Convert ms to seconds
        sample['Fwd IAT Mean'] = sample['Flow IAT Mean']
        sample['Fwd IAT Std'] = sample['Flow IAT Std']
        sample['Fwd IAT Max'] = sample['Flow IAT Max']
        sample['Fwd IAT Min'] = sample['Flow IAT Min']
        sample['Bwd IAT Total'] = sample['Flow Duration'] / 1000 * 0.9
        sample['Bwd IAT Mean'] = sample['Flow IAT Mean'] * 1.1
        sample['Bwd IAT Std'] = sample['Flow IAT Std'] * 1.1
        sample['Bwd IAT Max'] = sample['Flow IAT Max'] * 1.1
        sample['Bwd IAT Min'] = sample['Flow IAT Min'] * 1.1
        
        sample['Fwd PSH Flags'] = random.randint(0, 2)
        sample['Fwd Header Length'] = random.randint(20, 40)
        sample['Bwd Header Length'] = random.randint(20, 40)
        sample['Fwd Packets/s'] = sample['Total Fwd Packets'] / (sample['Flow Duration'] / 1000)
        sample['Bwd Packets/s'] = sample['Total Backward Packets'] / (sample['Flow Duration'] / 1000)
        sample['Packet Length Max'] = max(sample['Fwd Packet Length Max'], sample['Bwd Packet Length Max'])
        sample['Packet Length Mean'] = (sample['Fwd Packet Length Mean'] + sample['Bwd Packet Length Mean']) / 2
        sample['Packet Length Std'] = (sample['Fwd Packet Length Std'] + sample['Bwd Packet Length Std']) / 2
        sample['Packet Length Variance'] = sample['Packet Length Std'] ** 2
        sample['SYN Flag Count'] = 1 if random.random() > 0.8 else 0
        sample['URG Flag Count'] = 0
        sample['Avg Packet Size'] = (sample['Fwd Packets Length Total'] + sample['Bwd Packets Length Total']) / (sample['Total Fwd Packets'] + sample['Total Backward Packets'])
        sample['Avg Fwd Segment Size'] = sample['Fwd Packets Length Total'] / sample['Total Fwd Packets'] if sample['Total Fwd Packets'] > 0 else 0
        sample['Avg Bwd Segment Size'] = sample['Bwd Packets Length Total'] / sample['Total Backward Packets'] if sample['Total Backward Packets'] > 0 else 0
        
        sample['Subflow Fwd Packets'] = sample['Total Fwd Packets']
        sample['Subflow Fwd Bytes'] = sample['Fwd Packets Length Total']
        sample['Subflow Bwd Packets'] = sample['Total Backward Packets']
        sample['Subflow Bwd Bytes'] = sample['Bwd Packets Length Total']
        
        sample['Init Fwd Win Bytes'] = random.choice([8192, 16384, 65535])
        sample['Init Bwd Win Bytes'] = random.choice([8192, 16384, 65535])
        sample['Fwd Act Data Packets'] = sample['Total Fwd Packets']
        sample['Fwd Seg Size Min'] = random.randint(40, 100)
        
        sample['Active Mean'] = random.uniform(1, 10)
        sample['Active Std'] = random.uniform(0.5, 3)
        sample['Active Max'] = sample['Active Mean'] + sample['Active Std'] * 2
        sample['Active Min'] = max(0, sample['Active Mean'] - sample['Active Std'] * 2)
        
        sample['Idle Mean'] = random.uniform(5, 20)
        sample['Idle Std'] = random.uniform(1, 5)
        sample['Idle Max'] = sample['Idle Mean'] + sample['Idle Std'] * 2
        sample['Idle Min'] = max(0, sample['Idle Mean'] - sample['Idle Std'] * 2)
        
        sample['timestamp'] = datetime.now().isoformat()
        sample['is_anomaly'] = is_anomaly
        
        data.append(sample)
    
    return pd.DataFrame(data)

def stream_data_to_api(api_url="http://localhost:5000", interval=3, batch_size=5):
    """
    Stream generated data to detection API
    """
    print("Starting realistic network data generation...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            # Generate batch of data
            df_batch = generate_realistic_network_traffic(batch_size, anomaly_ratio=0.3)
            
            for _, row in df_batch.iterrows():
                # Prepare features for API (exclude metadata)
                features = {col: row[col] for col in df_batch.columns 
                           if col not in ['timestamp', 'is_anomaly']}
                
                # Send to detection API
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"features": features},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        status_color = "\033[91m" if result['prediction'] == 1 else "\033[92m"  # Red for attack, green for benign
                        reset_color = "\033[0m"
                        
                        print(f"{status_color}Prediction: {result['prediction']} ({result['status']}), "
                              f"Confidence: {result['confidence']:.3f}, "
                              f"Actual: {row['is_anomaly']}{reset_color}")
                    else:
                        print(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    print(f"Request failed: {e}")
            
            print("-" * 50)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nData generation stopped")

if __name__ == "__main__":
    stream_data_to_api()