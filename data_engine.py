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


def _ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _adx(high, low, close, length=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)

    atr = _atr(high, low, close, length)
    plus_dm_smooth = plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / length, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return plus_di, minus_di, adx


def _supertrend(df, length=10, multiplier=3):
    high, low, close = df['High'], df['Low'], df['Close']
    atr = _atr(high, low, close, length)
    hl2 = (high + low) / 2
    upperband = (hl2 + multiplier * atr).copy()
    lowerband = (hl2 - multiplier * atr).copy()

    direction = pd.Series(1, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        if i == 0:
            direction.iloc[i] = 1
            supertrend.iloc[i] = lowerband.iloc[i]
            continue

        if close.iloc[i] > upperband.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        supertrend.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]

    return supertrend, direction


def _macd(close, fast=12, slow=26, signal=9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _donchian(high, low, length=20):
    # Kendi gunu haric onceki 'length' gunun en yuksek/dusugu - bugunun mumuyla
    # karsilastirildiginda lookahead-bias olmamasi icin shift(1) ile kaydiriliyor.
    ust = high.rolling(length).max().shift(1)
    alt = low.rolling(length).min().shift(1)
    return ust, alt


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI_14'] = _rsi(df['Close'], 14)

    macd_line, signal_line, hist = _macd(df['Close'], 12, 26, 9)
    df['MACD_12_26_9'] = macd_line
    df['MACDs_12_26_9'] = signal_line
    df['MACDh_12_26_9'] = hist

    df['EMA_50'] = _ema(df['Close'], 50)
    df['EMA_200'] = _ema(df['Close'], 200)

    supertrend, direction = _supertrend(df, length=10, multiplier=3)
    df['SUPERT_10_3'] = supertrend
    df['SUPERTd_10_3'] = direction

    donchian_ust, donchian_alt = _donchian(df['High'], df['Low'], 20)
    df['DONCHIAN_UST_20'] = donchian_ust
    df['DONCHIAN_ALT_20'] = donchian_alt

    plus_di, minus_di, adx = _adx(df['High'], df['Low'], df['Close'], 14)
    df['DMP_14'] = plus_di
    df['DMN_14'] = minus_di
    df['ADX_14'] = adx

    df['ATR_14'] = _atr(df['High'], df['Low'], df['Close'], 14)

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

        print(f"[{ticker}] indikatorler hesaplaniyor...")
        df = add_indicators(df)

        print(f"[{ticker}] veritabanina kaydediliyor...")
        save_to_db(df, ticker, conn)

    conn.close()
    print(f"\nTamamlandi. Veriler '{DB_PATH}' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
