CREATE TABLE IF NOT EXISTS fraud_scores (
    tid         BIGINT PRIMARY KEY,
    score       DOUBLE PRECISION,
    fraud       BOOLEAN,
    ts          TIMESTAMPTZ DEFAULT NOW()
);