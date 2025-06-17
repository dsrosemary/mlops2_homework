import six, sys
sys.modules["kafka.vendor.six"] = six
sys.modules["kafka.vendor.six.moves"] = six.moves

import os
import json
import joblib
import pandas as pd
import time
from kafka.errors import NoBrokersAvailable
from kafka import KafkaConsumer, KafkaProducer
from catboost import CatBoostClassifier
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

from src.preprocessing import transform_preprocessor, MODEL_FEATURES
from src.scorer import load_model   

BROKER       = os.getenv("KAFKA_BROKER", "kafka:9092")
IN_TOPIC     = os.getenv("INPUT_TOPIC", "transactions")
OUT_TOPIC    = os.getenv("OUTPUT_TOPIC", "scores")
THRESHOLD    = float(os.getenv("THRESHOLD", 0.98))

for _ in range(30):           
    try:
        consumer = KafkaConsumer(
            IN_TOPIC,
            bootstrap_servers=BROKER,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
        )
        break                  
    except NoBrokersAvailable:
        time.sleep(1)
else:
    raise RuntimeError("Kafka broker is still unavailable after 30 s")

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda m: json.dumps(m).encode("utf-8"),
)
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=BROKER,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
)

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda m: json.dumps(m).encode("utf-8"),
)

helpers = joblib.load("models/preprocessor.pkl")
model   = load_model("models/my_catboost.cbm")  

for msg in consumer:
    rec = msg.value                
    df  = pd.DataFrame([rec])
    X   = transform_preprocessor(df, helpers)[MODEL_FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    flag  = int(proba > THRESHOLD)
    out   = {
        "transaction_id": rec.get("index", rec.get("transaction_id")),
        "score": proba,
        "fraud_flag": flag,
    }
    producer.send(OUT_TOPIC, out)