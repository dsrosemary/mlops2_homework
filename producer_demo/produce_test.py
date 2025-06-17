import six, sys                  
sys.modules["kafka.vendor.six"] = six
sys.modules["kafka.vendor.six.moves"] = six.moves

from kafka import KafkaProducer
import pandas as pd, json, argparse
import os, argparse
ap = argparse.ArgumentParser()
ap.add_argument("csv_path")
ap.add_argument("--broker", default=os.getenv("KAFKA_BROKER", "localhost:9092"))
args = ap.parse_args()
BROKER = args.broker       
TOPIC  = "transactions"

ap = argparse.ArgumentParser()
ap.add_argument("csv_path")
args = ap.parse_args()

df = pd.read_csv(args.csv_path)
producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda m: json.dumps(m).encode("utf-8"),
)

for i, row in df.iterrows():
    payload = row.to_dict()
    payload["index"] = int(i)
    producer.send(TOPIC, payload).get(10)   
    print("sent", i)

producer.flush()