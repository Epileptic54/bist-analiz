import sqlite3

import pandas as pd
import pandas_ta as ta
import yfinance as yf

DB_PATH = "bist_portfolio.db"
TICKERS = ["ASELS.IS", "ASTOR.IS", "THYAO.IS", "BIMAS.IS", "AKBNK.IS"]


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="1y", interval="1d")
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.supertrend(length=10, multiplier=3, append=True)
    df.ta.stochrsi(append=True)
    return df


def save_to_db(df: pd.DataFrame, ticker: str, conn: sqlite3.Connection) -> None:
    table_name = ticker.replace(".", "_")
    df = df.reset_index()
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    for ticker in TICKERS:
        print(f"[{ticker}] veri cekiliyor...")
        df = fetch_data(ticker)

        if df.empty:
            print(f"[{ticker}] veri bulunamadi, atlaniyor.")
            continue

        print(f"[{ticker}] indikatorler hesaplaniyor...")
        df = add_indicators(df)

        print(f"[{ticker}] veritabanina kaydediliyor...")
        save_to_db(df, ticker, conn)

    conn.close()
    print(f"\nTamamlandi. Veriler '{DB_PATH}' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
