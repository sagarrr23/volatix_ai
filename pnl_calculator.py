import pandas as pd

def simulate_trade_outcomes(
    file_path="C:/NEWW/volatix_ai/logs/backtest_results.csv",
    output_path="pnl_results.csv"
):
    df = pd.read_csv(file_path)

    # ✅ Ensure required columns exist
    required = ["timestamp", "signal", "entry", "sl", "tp", "size"]
    for col in required:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    # Add result columns
    df["pnl"] = 0.0
    df["result"] = "WAIT"

    for i in range(len(df) - 1):
        row = df.iloc[i]

        # 🚫 Skip if not a trade
        if row["signal"] not in ["BUY", "SELL"]:
            continue

        # Simulate next candle move
        next_row = df.iloc[i + 1]
        entry = row["entry"]
        sl = row["sl"]
        tp = row["tp"]
        size = row["size"]

        # 🔍 Check if SL/TP is missing
        if pd.isna(sl) or pd.isna(tp) or pd.isna(entry) or size == 0:
            df.at[i, "result"] = "INVALID"
            continue

        # 👀 Simulate using close/entry as proxy for move
        candle_high = max(next_row.get("entry", entry), entry)
        candle_low = min(next_row.get("entry", entry), entry)

        # Logic for BUY trade
        if row["signal"] == "BUY":
            if candle_high >= tp:
                pnl = (tp - entry) * size
                df.at[i, "pnl"] = pnl
                df.at[i, "result"] = "WIN"
            elif candle_low <= sl:
                pnl = (sl - entry) * size
                df.at[i, "pnl"] = pnl
                df.at[i, "result"] = "LOSS"
            else:
                df.at[i, "result"] = "OPEN"

        # Logic for SELL trade
        elif row["signal"] == "SELL":
            if candle_low <= tp:
                pnl = (entry - tp) * size
                df.at[i, "pnl"] = pnl
                df.at[i, "result"] = "WIN"
            elif candle_high >= sl:
                pnl = (entry - sl) * size
                df.at[i, "pnl"] = pnl
                df.at[i, "result"] = "LOSS"
            else:
                df.at[i, "result"] = "OPEN"

    # Save updated DataFrame
    df.to_csv(output_path, index=False)
    print(f"[✓] PnL calculation complete. Results saved to: {output_path}")

if __name__ == "__main__":
    simulate_trade_outcomes()
