import six, sys
sys.modules["kafka.vendor.six"] = six
sys.modules["kafka.vendor.six.moves"] = six.moves

import os, json, time, psycopg2
from kafka import KafkaConsumer, errors

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC  = os.getenv("SCORE_TOPIC",  "scores")
PG_DSN = os.getenv("PG_DSN", "postgres://fraud:fraud@postgres:5432/fraud_db")

for _ in range(30):                     
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BROKER,
            value_deserializer=lambda m: json.loads(m.decode()),
            auto_offset_reset="earliest",
        )
        break
    except errors.NoBrokersAvailable:
        time.sleep(1)
else:
    raise RuntimeError("Kafka broker not available")

conn = psycopg2.connect(PG_DSN)
cur  = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS fraud_scores(
                 tid   BIGINT PRIMARY KEY,
                 score DOUBLE PRECISION,
                 fraud BOOLEAN,
                 ts    TIMESTAMPTZ DEFAULT NOW())""")
conn.commit()

for msg in consumer:
    d = msg.value
    flag = bool(d["fraud_flag"])         
    cur.execute(
        "INSERT INTO fraud_scores (tid, score, fraud) "
        "VALUES (%s, %s, %s) ON CONFLICT (tid) DO NOTHING",
        (d["transaction_id"], d["score"], flag),
    )
    conn.commit()