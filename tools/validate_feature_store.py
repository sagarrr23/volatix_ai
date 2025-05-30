import duckdb
import pandas as pd
import os

DB_PATH = "feature_store/btc_features.duckdb"
CLEAN_PATH = "feature_store/btc_features_clean.duckdb"

def validate_table(conn, table):
    print(f"\n🔍 Checking table: {table}")
    df = conn.execute(f"SELECT * FROM {table} LIMIT 5").df()
    print("📌 Columns:")
    print(df.columns)
    print("\n🧾 Sample Data:")
    print(df.head())


    # Align check
    if {'y_direction', 'y_reward', 'y_confidence'}.issubset(df.columns):
        direction_na = df['y_direction'].isna().sum()
        reward_na = df['y_reward'].isna().sum()
        conf_na = df['y_confidence'].isna().sum()
        print(f"📊 Targets — direction: {direction_na}, reward: {reward_na}, confidence: {conf_na}")
    else:
        print("⚠️ Missing target columns.")

    return df

def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Feature store not found at {DB_PATH}")
        return

    conn = duckdb.connect(DB_PATH)
    tables = conn.execute("SHOW TABLES").fetchall()
    print("📋 Tables found:", [t[0] for t in tables])

    for t in tables:
        validate_table(conn, t[0])

    print("\n✅ Done. If needed, clean tables and export to new db.")

if __name__ == "__main__":
    main()
