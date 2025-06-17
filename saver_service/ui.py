import os
import psycopg2
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PG_DSN = os.getenv("PG_DSN", "postgres://fraud:fraud@postgres:5432/fraud_db")
conn   = psycopg2.connect(PG_DSN)

st.title("Fraud Scores Dashboard")

if st.button("Посмотреть результаты"):
    fraud_df = pd.read_sql(
        "SELECT tid, score, ts FROM fraud_scores WHERE fraud ORDER BY ts DESC LIMIT 10",
        conn,
    )
    st.subheader("Последние подозрительные транзакции")
    st.table(fraud_df)

    scores_df = pd.read_sql(
        "SELECT score FROM fraud_scores ORDER BY ts DESC LIMIT 100",
        conn,
    )
    if not scores_df.empty:
        st.subheader("Распределение скоров (последние 100)")
        plt.figure()
        scores_df["score"].hist(bins=20)
        st.pyplot(plt)