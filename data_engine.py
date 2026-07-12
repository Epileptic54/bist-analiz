import sqlite3

import pandas as pd
import yfinance as yf

DB_PATH = "bist_portfolio.db"
TICKERS = ["ASELS.IS", "ASTOR.IS", "THYAO.IS", "BIMAS.IS", "AKBNK.IS"]


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="2y", interval="1d")
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
    df = df.dropna(subset=["Close"])
    return df


def save_to_db(df: pd.DataFrame, ticker: str, conn: sqlite3.Connection) -> None:
    table_name = ticker.replace(".", "_")
    df = df.reset_index()
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def init_ekstra_tablolar(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS watchlist (ticker TEXT PRIMARY KEY, ad TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS portfoy (ticker TEXT PRIMARY KEY, adet REAL NOT NULL, maliyet REAL NOT NULL)")
    conn.commit()

    sayi = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
    if sayi == 0:
        varsayilan = [
            ("ASELS_IS", "Aselsan"),
            ("ASTOR_IS", "Astor Enerji"),
            ("THYAO_IS", "Türk Hava Yolları"),
            ("BIMAS_IS", "BİM Mağazalar"),
            ("AKBNK_IS", "Akbank"),
        ]
        conn.executemany("INSERT INTO watchlist (ticker, ad) VALUES (?, ?)", varsayilan)
        conn.commit()


def watchlist_getir(conn: sqlite3.Connection) -> dict:
    satirlar = conn.execute("SELECT ticker, ad FROM watchlist ORDER BY ad").fetchall()
    return {t: a for t, a in satirlar}


def watchlist_ekle(conn: sqlite3.Connection, ticker_db: str, ad: str) -> None:
    conn.execute("INSERT OR REPLACE INTO watchlist (ticker, ad) VALUES (?, ?)", (ticker_db, ad))
    conn.commit()


def watchlist_sil(conn: sqlite3.Connection, ticker_db: str) -> None:
    conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker_db,))
    conn.execute("DELETE FROM portfoy WHERE ticker = ?", (ticker_db,))
    conn.commit()


def portfoy_getir(conn: sqlite3.Connection) -> dict:
    satirlar = conn.execute("SELECT ticker, adet, maliyet FROM portfoy").fetchall()
    return {t: {"adet": a, "maliyet": m} for t, a, m in satirlar}


def portfoy_kaydet(conn: sqlite3.Connection, ticker_db: str, adet: float, maliyet: float) -> None:
    conn.execute("INSERT OR REPLACE INTO portfoy (ticker, adet, maliyet) VALUES (?, ?, ?)", (ticker_db, adet, maliyet))
    conn.commit()


def portfoy_sil(conn: sqlite3.Connection, ticker_db: str) -> None:
    conn.execute("DELETE FROM portfoy WHERE ticker = ?", (ticker_db,))
    conn.commit()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    for ticker in TICKERS:
        print(f"[{ticker}] veri cekiliyor...")
        df = fetch_data(ticker)

        if df.empty:
            print(f"[{ticker}] veri bulunamadi, atlaniyor.")
            continue

        print(f"[{ticker}] veritabanina kaydediliyor...")
        save_to_db(df, ticker, conn)

    conn.close()
    print(f"\nTamamlandi. Veriler '{DB_PATH}' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
