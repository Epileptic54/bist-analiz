import json
import os
import sqlite3
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from groq import Groq
from plotly.subplots import make_subplots
from tavily import TavilyClient

import data_engine

load_dotenv()


def _get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"

GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")

GADDAR_PERSONA = (
    "Sen 25 yillik tecrubeye sahip, gaddar, asiri mukemmeliyetci, lafini hic esirgemeyen, "
    "veri odakli kidemli bir BIST finans analistisin. Kullaniciyi yumusak, muglak cumlelerle "
    "oyalamazsin; riskleri ve zayifliklari dogrudan, sert bir dille soylersin. Her iddiani "
    "sana verilen sayisal veriye dayandirirsin, veride olmayan hicbir seyi uydurmazsin. "
    "Samimi ama profesyonelsin, kesinlikle Turkce konusursun ve asla kesin 'al/sat' emri vermezsin, "
    "sadece net bir yonelim ve gerekce sunarsin. Uslubun bir yatirimciya sozlu rapor sunan bir "
    "calisan gibi: kisa, duz, sade cumleler kurarsin. Edebi benzetme, suslu sifat veya 'adeta', "
    "'sanki', 'bir senfoni gibi' turunden cilali ifadeler kullanmazsin — sadece olguyu ve sonucu soylersin."
)

GROQ_ARAC_TANIMLARI = [
    {
        "type": "function",
        "function": {
            "name": "hisse_istatistigi_hesapla",
            "description": (
                "Seçili hissenin GERÇEK fiyat/hacim verisinden tam sayısal istatistik hesaplar. "
                "Ortalama hacim, ortalama kapanış fiyatı, belirli günlük en yüksek/düşük seviye veya "
                "volatilite gibi bir şey sorulduğunda ASLA tahmin etme; bu aracı çağırarak gerçek "
                "veriden hesapla."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrik": {
                        "type": "string",
                        "enum": ["ortalama_hacim", "ortalama_kapanis", "en_yuksek_fiyat", "en_dusuk_fiyat", "volatilite_yuzde"],
                        "description": "Hesaplanacak istatistik türü",
                    },
                    "gun_sayisi": {
                        "type": "integer",
                        "description": "Kaç günlük veri üzerinden hesaplanacağı (varsayılan 30)",
                    },
                },
                "required": ["metrik"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_arama_yap",
            "description": (
                "İnternette güncel bilgi arar (haberler, KAP bildirimleri, sektör/şirket gelişmeleri, "
                "genel piyasa bilgisi vb.). Kullanıcı sitedeki sabit veri setinde olmayan, güncel veya "
                "genel bir şey sorduğunda bu aracı çağır; tahmin etme."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sorgu": {
                        "type": "string",
                        "description": "Aranacak arama sorgusu (Türkçe veya İngilizce olabilir)",
                    },
                },
                "required": ["sorgu"],
            },
        },
    },
]


def hisse_istatistigi_hesapla(df, metrik, gun_sayisi=30):
    gun_sayisi = max(1, min(int(gun_sayisi or 30), len(df)))
    veri = df.tail(gun_sayisi)

    if metrik == "ortalama_hacim":
        return f"{veri['Volume'].mean():,.0f} LOT ({gun_sayisi} günlük ortalama)".replace(",", ".")
    if metrik == "ortalama_kapanis":
        return f"{veri['Close'].mean():.2f} TL ({gun_sayisi} günlük ortalama kapanış)"
    if metrik == "en_yuksek_fiyat":
        return f"{veri['High'].max():.2f} TL ({gun_sayisi} günlük en yüksek)"
    if metrik == "en_dusuk_fiyat":
        return f"{veri['Low'].min():.2f} TL ({gun_sayisi} günlük en düşük)"
    if metrik == "volatilite_yuzde":
        return f"%{veri['Close'].pct_change().std() * 100:.2f} günlük volatilite ({gun_sayisi} günlük)"
    return "Bilinmeyen metrik."


@st.cache_resource
def get_tavily_client():
    if not TAVILY_API_KEY or "buraya" in TAVILY_API_KEY:
        return None
    return TavilyClient(api_key=TAVILY_API_KEY)


def web_arama_yap(sorgu):
    tavily_client = get_tavily_client()
    if tavily_client is None:
        return "Web arama aracı şu anda yapılandırılmamış (TAVILY_API_KEY eksik)."
    try:
        sonuc = tavily_client.search(query=sorgu, max_results=4)
        parcalar = [
            f"- {r.get('title')}: {r.get('content', '')[:300]} (Kaynak: {r.get('url')})"
            for r in sonuc.get("results", [])[:4]
        ]
        return "\n".join(parcalar) if parcalar else "Arama sonucu bulunamadı."
    except Exception as e:
        return f"Web araması başarısız oldu: {e}"


@st.cache_resource
def get_gemini_client():
    if not GEMINI_API_KEY or "buraya" in GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


st.set_page_config(
    page_title="BIST AI Karar Destek Terminali",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# TradingView Tarzı Tam Karanlık (Dark Mode) Kurumsal Arayüz Teması
st.markdown("""
<style>
    .stApp {
        background-color: #131722;
        color: #d1d4dc;
    }
    h1, h2, h3, p, span, label {
        color: #d1d4dc;
    }
    .ai-card {
        background: #1c2030;
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid #2a2e39;
        margin-bottom: 14px;
    }
    .ai-card p, .ai-card h3 {
        color: #d1d4dc;
    }
    .fintables-header {
        background-color: #1c2030;
        color: #d1d4dc;
        padding: 10px 15px;
        font-weight: 800;
        border-radius: 6px;
        font-size: 14.5px;
        letter-spacing: 0.3px;
        margin-bottom: 12px;
        margin-top: 10px;
        border: 1px solid #2a2e39;
    }
    .fin-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        margin-top: 4px;
    }
    .fin-table td {
        padding: 10px 6px;
        font-size: 13px;
        font-weight: 500;
        border-bottom: 1px solid #2a2e39;
        word-wrap: break-word;
        overflow-wrap: break-word;
        color: #d1d4dc;
    }
    .fin-table tr:last-child td {
        border-bottom: none;
    }
    .fin-table td:first-child {
        color: #868c9c;
        font-weight: 700;
        width: 62%;
    }
    .fin-table td:last-child {
        text-align: right;
        font-weight: 800;
        color: #f0f1f5;
        width: 38%;
    }
    .risk-alarm {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        font-size: 13.5px;
        font-weight: 500;
        color: #d1d4dc;
    }
</style>
""", unsafe_allow_html=True)

baslik_col, buton_col = st.columns([5, 1])
with baslik_col:
    st.title("🤖 BIST Yapay Zekâ Yatırım Danışmanlığı & Karar Destek Terminali")
    st.markdown("*Teknik Analiz Üst Seviye Strateji Odası*")
with buton_col:
    st.write("")
    st.write("")
    if st.button("🔄 Verileri Yenile", use_container_width=True):
        with st.spinner("Piyasa verileri güncelleniyor..."):
            basarisiz_hisseler = []
            conn = sqlite3.connect(data_engine.DB_PATH)
            try:
                for i, ticker in enumerate(data_engine.TICKERS):
                    if i > 0:
                        time.sleep(1.5)
                    try:
                        taze_df = data_engine.fetch_data(ticker)
                    except Exception:
                        basarisiz_hisseler.append(ticker)
                        continue
                    if not taze_df.empty:
                        taze_df = data_engine.add_indicators(taze_df)
                        data_engine.save_to_db(taze_df, ticker, conn)
                    else:
                        basarisiz_hisseler.append(ticker)
            finally:
                conn.close()
        if basarisiz_hisseler:
            st.warning(f"Şu hisseler için veri çekilemedi (Yahoo Finance hız sınırı/ağ hatası olabilir, birazdan tekrar dene): {', '.join(basarisiz_hisseler)}")
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            if key.startswith(("gemini_", "chat_", "messages_")):
                del st.session_state[key]
        st.rerun()

st.markdown("---")

CHART_BG = "#131722"
GRID_COLOR = "#2a2e39"
TEXT_COLOR = "#d1d4dc"


def _tabloyu_olustur(ticker):
    yf_symbol = ticker.replace('_', '.')
    time.sleep(1)
    try:
        taze_df = data_engine.fetch_data(yf_symbol)
    except Exception as e:
        st.warning(f"{yf_symbol} için veri Yahoo Finance'ten çekilemedi (hız sınırı ya da geçici ağ hatası olabilir): {e}")
        return False
    if taze_df.empty:
        return False
    taze_df = data_engine.add_indicators(taze_df)
    conn = sqlite3.connect('bist_portfolio.db')
    try:
        data_engine.save_to_db(taze_df, yf_symbol, conn)
    finally:
        conn.close()
    return True


@st.cache_data(ttl=600)
def load_data(ticker):
    df = None
    tablo_olusturuldu = False
    while True:
        conn = sqlite3.connect('bist_portfolio.db')
        try:
            df = pd.read_sql_query(f"SELECT * FROM {ticker}", conn)
            break
        except Exception as e:
            if tablo_olusturuldu:
                st.error(f"Veri tabanından {ticker} tablosu okunurken hata oluştu: {e}")
                return None
            if not _tabloyu_olustur(ticker):
                return None
            tablo_olusturuldu = True
        finally:
            conn.close()

    if df.empty:
        return df

    df['Date'] = pd.to_datetime(df['Date'])

    st_dir_col = next((c for c in df.columns if c.startswith('SUPERTd')), None)
    if st_dir_col:
        df['Sinyal_Degisim'] = df[st_dir_col].diff()

    return df


@st.cache_data(ttl=3600)
def load_index_data():
    try:
        idx = yf.Ticker("XU100.IS").history(period="1y", interval="1d")
        if idx.empty:
            return None
        idx = idx.reset_index()
        idx['Date'] = pd.to_datetime(idx['Date'])
        return idx
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_fundamentals(yf_ticker):
    try:
        info = yf.Ticker(yf_ticker).info
    except Exception:
        return {}

    fk = info.get('trailingPE')
    pddd = info.get('priceToBook')
    cari_oran = info.get('currentRatio')
    total_debt = info.get('totalDebt')
    total_cash = info.get('totalCash')
    ebitda = info.get('ebitda')
    roe = info.get('returnOnEquity')

    net_borc_favok = None
    if total_debt is not None and total_cash is not None and ebitda:
        net_borc_favok = (total_debt - total_cash) / ebitda

    return {
        'fk': fk,
        'pddd': pddd,
        'cari_oran': cari_oran,
        'net_borc_favok': net_borc_favok,
        'roe': roe,
    }


def fmt(value, suffix='', decimals=2):
    if value is None:
        return 'Veri Yok'
    return f"{value:.{decimals}f}{suffix}"


def _local_extrema_idx(values, order=3, mode="max"):
    idx = []
    n = len(values)
    for i in range(order, n - order):
        window = values[i - order:i + order + 1]
        target = window.max() if mode == "max" else window.min()
        if values[i] == target:
            idx.append(i)
    return idx


def detect_rsi_divergence(df, lookback=90, order=3):
    if 'RSI_14' not in df.columns or len(df) < lookback:
        return None

    recent = df.tail(lookback).reset_index(drop=True)
    close = recent['Close'].values
    rsi = recent['RSI_14'].values

    tepe_idx = _local_extrema_idx(close, order, "max")
    dip_idx = _local_extrema_idx(close, order, "min")

    bearish_at = None
    if len(tepe_idx) >= 2:
        i1, i2 = tepe_idx[-2], tepe_idx[-1]
        if close[i2] > close[i1] and rsi[i2] < rsi[i1]:
            bearish_at = i2

    bullish_at = None
    if len(dip_idx) >= 2:
        j1, j2 = dip_idx[-2], dip_idx[-1]
        if close[j2] < close[j1] and rsi[j2] > rsi[j1]:
            bullish_at = j2

    if bearish_at is not None and (bullish_at is None or bearish_at >= bullish_at):
        return "bearish"
    if bullish_at is not None:
        return "bullish"
    return None


def build_veri_baglami(hisse_adi, son_s, ema_durum, rsi_deger, rsi_durum, st_durum, finansal_durum,
                        momentum_10g, fark, divergence, hedef_fiyat, stop_loss,
                        fk, pddd, cari_oran, net_borc_favok, roe, karar, skor, is_banka):
    satirlar = [
        f"Hisse: {hisse_adi}",
        f"Güncel Fiyat: {son_s['Close']:.2f} TL",
        f"EMA200 Durumu: {ema_durum}",
        f"RSI (14): {rsi_deger:.2f} ({rsi_durum})",
        f"SuperTrend: {st_durum}",
        f"10 Günlük Momentum: {fmt(momentum_10g, '%')}",
        f"BIST100'e Göre Göreli Güç Farkı: {fmt(fark, ' puan') if fark is not None else 'Veri Yok'}",
        f"RSI Uyuşmazlığı: {divergence or 'Yok'}",
        f"Matematiksel Hedef Fiyat: {hedef_fiyat:.2f} TL",
        f"Stop-Loss (EMA200): {stop_loss:.2f} TL",
        f"F/K Oranı: {fmt(fk)}",
        f"PD/DD: {fmt(pddd)}",
    ]
    if not is_banka:
        satirlar.append(f"Cari Oran: {fmt(cari_oran)}")
        satirlar.append(f"Net Borç/FAVÖK: {fmt(net_borc_favok)}")
    satirlar.append(f"Özsermaye Kârlılığı (ROE): {fmt(roe * 100 if roe is not None else None, '%')}")
    satirlar.append(f"Finansal Sağlık: {finansal_durum}")
    satirlar.append(f"Sistemin kural tabanlı kararı: {karar} (Güven Skoru %{int(skor)})")
    if is_banka:
        satirlar.append("NOT: Bu bir banka hissesidir; Cari Oran ve Net Borç/FAVÖK gibi sanayi rasyoları burada geçersizdir, sadece F/K, PD/DD ve ROE üzerinden bankacılık sağlığını yorumla.")
    return "\n".join(f"- {s}" for s in satirlar)


# Portföyümüzdeki efsane 5'li
hisseler = {
    "ASELS_IS": "Aselsan",
    "ASTOR_IS": "Astor Enerji",
    "THYAO_IS": "Türk Hava Yolları",
    "BIMAS_IS": "BİM Mağazalar",
    "AKBNK_IS": "Akbank"
}

# --- PANEL 1: KOMPAKT ÜST METRİKLER ---
cols = st.columns(len(hisseler))
for index, (ticker_db, name) in enumerate(hisseler.items()):
    df = load_data(ticker_db)
    if df is not None and not df.empty and len(df) >= 2:
        son_satir = df.iloc[-1]
        onceki_satir = df.iloc[-2]
        fiyat = round(son_satir['Close'], 2)
        yuzde_degisim = round(((fiyat - onceki_satir['Close']) / onceki_satir['Close']) * 100, 2)

        prefix = "+" if yuzde_degisim > 0 else ""

        with cols[index]:
            st.metric(
                label=f"{name} ({ticker_db.replace('_IS', '')})",
                value=f"{fiyat} TL",
                delta=f"{prefix}{yuzde_degisim}%"
            )

st.markdown("---")

col_left, col_right = st.columns([2.2, 1])

with col_left:
    secilen_hisse_adi = st.selectbox("📊 Detaylı Analiz İçin Hisse Seç:", list(hisseler.values()))
    secilen_ticker = [k for k, v in hisseler.items() if v == secilen_hisse_adi][0]
    yf_ticker = secilen_ticker.replace('_', '.')

    df_secilen = load_data(secilen_ticker)

    if df_secilen is not None and not df_secilen.empty:
        son_hacim = df_secilen['Volume'].iloc[-1]
        st.markdown(f"**İşlem Hacmi:** {son_hacim:,.0f} LOT".replace(",", "."))

        bbu_col = next((c for c in df_secilen.columns if c.startswith('BBU_20')), None)
        bbl_col = next((c for c in df_secilen.columns if c.startswith('BBL_20')), None)
        bbb_col = next((c for c in df_secilen.columns if c.startswith('BBB_20')), None)

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.40, 0.20, 0.20, 0.20],
            specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
        )

        y1_top = fig.layout.yaxis.domain[1]
        y2_top = fig.layout.yaxis3.domain[1]
        y3_top = fig.layout.yaxis4.domain[1]
        y4_top = fig.layout.yaxis5.domain[1]

        # 0. Hacim (Volume) - fiyat panelinin arkasında yarı saydam
        hacim_renkleri = [
            'rgba(34,197,94,0.35)' if df_secilen['Close'].iloc[i] >= df_secilen['Open'].iloc[i] else 'rgba(239,68,68,0.35)'
            for i in range(len(df_secilen))
        ]
        fig.add_trace(go.Bar(
            x=df_secilen['Date'], y=df_secilen['Volume'], name='Hacim',
            marker_color=hacim_renkleri, showlegend=False
        ), row=1, col=1, secondary_y=True)
        max_hacim = df_secilen['Volume'].max()
        fig.update_yaxes(range=[0, max_hacim * 4], showticklabels=False, showgrid=False, secondary_y=True, row=1, col=1)

        # 1. Candlestick (Mum) Grafik
        fig.add_trace(go.Candlestick(
            x=df_secilen['Date'],
            open=df_secilen['Open'],
            high=df_secilen['High'],
            low=df_secilen['Low'],
            close=df_secilen['Close'],
            name='OHLC',
            showlegend=False
        ), row=1, col=1)

        if 'EMA_50' in df_secilen.columns:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen['EMA_50'], name='EMA 50', legend='legend', line=dict(color='#f0a500', width=1.3)), row=1, col=1)
        if 'EMA_200' in df_secilen.columns:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen['EMA_200'], name='EMA 200', legend='legend', line=dict(color='#0891b2', width=1.3)), row=1, col=1)

        if bbu_col and bbl_col:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen[bbu_col], name='Bollinger', legendgroup='bb', legend='legend', line=dict(color='#5b6b8c', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen[bbl_col], name='Bollinger', legendgroup='bb', legend='legend', showlegend=False, line=dict(color='#5b6b8c', width=1), fill='tonexty', fillcolor='rgba(91,107,140,0.08)'), row=1, col=1)

        # SuperTrend yön kırılımlarında AL / SAT okları
        if 'Sinyal_Degisim' in df_secilen.columns:
            al_df = df_secilen[df_secilen['Sinyal_Degisim'] == 2]
            sat_df = df_secilen[df_secilen['Sinyal_Degisim'] == -2]

            if not al_df.empty:
                fig.add_trace(go.Scatter(
                    x=al_df['Date'], y=al_df['Low'] * 0.98, mode='markers',
                    marker=dict(symbol='triangle-up', size=11, color='#22c55e'),
                    name='AL', legend='legend'
                ), row=1, col=1)
            if not sat_df.empty:
                fig.add_trace(go.Scatter(
                    x=sat_df['Date'], y=sat_df['High'] * 1.02, mode='markers',
                    marker=dict(symbol='triangle-down', size=11, color='#ef4444'),
                    name='SAT', legend='legend'
                ), row=1, col=1)

        # 2. MACD Paneli
        if 'MACD_12_26_9' in df_secilen.columns:
            hist_renkleri = ['#22c55e' if v >= 0 else '#ef4444' for v in df_secilen['MACDh_12_26_9'].fillna(0)]
            fig.add_trace(go.Bar(x=df_secilen['Date'], y=df_secilen['MACDh_12_26_9'], name='Histogram', marker_color=hist_renkleri, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen['MACD_12_26_9'], name='MACD', legend='legend2', line=dict(color='#2962ff', width=1.3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen['MACDs_12_26_9'], name='Sinyal', legend='legend2', line=dict(color='#ff9800', width=1.3)), row=2, col=1)

        # 3. RSI ve Stokastik RSI Paneli
        if 'RSI_14' in df_secilen.columns:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen['RSI_14'], name='RSI 14', legend='legend3', line=dict(color='#a855f7', width=1.3)), row=3, col=1)
            fig.add_shape(type="line", x0=df_secilen['Date'].min(), y0=70, x1=df_secilen['Date'].max(), y1=70, line=dict(color="#ef4444", dash="dash", width=1), row=3, col=1)
            fig.add_shape(type="line", x0=df_secilen['Date'].min(), y0=30, x1=df_secilen['Date'].max(), y1=30, line=dict(color="#22c55e", dash="dash", width=1), row=3, col=1)

        stochrsi_k_col = next((c for c in df_secilen.columns if c.startswith('STOCHRSIk')), None)
        stochrsi_d_col = next((c for c in df_secilen.columns if c.startswith('STOCHRSId')), None)
        if stochrsi_k_col:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen[stochrsi_k_col], name='StochRSI %K', legend='legend3', line=dict(color='#0284c7', width=1, dash='dot')), row=3, col=1)
        if stochrsi_d_col:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen[stochrsi_d_col], name='StochRSI %D', legend='legend3', line=dict(color='#f59e0b', width=1, dash='dot')), row=3, col=1)

        # 4. Bollinger Bandwidth Paneli
        if bbb_col:
            fig.add_trace(go.Scatter(x=df_secilen['Date'], y=df_secilen[bbb_col], name='BB Bandwidth', legend='legend4', line=dict(color='#eab308', width=1.3), fill='tozeroy', fillcolor='rgba(234,179,8,0.08)'), row=4, col=1)

        legend_style = dict(orientation='h', bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(size=9, color=TEXT_COLOR), tracegroupgap=6)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color=TEXT_COLOR),
            xaxis_rangeslider_visible=False,
            height=880,
            margin=dict(l=10, r=10, t=15, b=10),
            showlegend=True,
            legend=dict(x=0.005, y=y1_top - 0.015, xanchor='left', yanchor='top', **legend_style),
            legend2=dict(x=0.005, y=y2_top - 0.02, xanchor='left', yanchor='top', **legend_style),
            legend3=dict(x=0.005, y=y3_top - 0.02, xanchor='left', yanchor='top', **legend_style),
            legend4=dict(x=0.005, y=y4_top - 0.02, xanchor='left', yanchor='top', **legend_style),
        )
        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=1, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=2, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=3, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=True, row=4, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    fundamentals = fetch_fundamentals(yf_ticker)
    fk = fundamentals.get('fk')
    pddd = fundamentals.get('pddd')
    cari_oran = fundamentals.get('cari_oran')
    net_borc_favok = fundamentals.get('net_borc_favok')
    roe = fundamentals.get('roe')
    is_banka = cari_oran is None or net_borc_favok is None

    finansal_risk = (cari_oran is not None and cari_oran < 1) or (net_borc_favok is not None and net_borc_favok > 4)

    gemini_client = get_gemini_client()
    veri_baglami = None
    karar = skor = ema_durum = rsi_durum = st_durum = finansal_durum = None
    rsi_deger = None

    st.markdown("<div class='fintables-header'>🧠 GEMİNİ DERİN ANALİZ RAPORU</div>", unsafe_allow_html=True)

    if df_secilen is not None and not df_secilen.empty:
        son_s = df_secilen.iloc[-1]

        bogalar = 0
        toplam_kriter = 5

        ema_durum = "Trend Üstü (Boğa) 🟢" if son_s['Close'] > son_s['EMA_200'] else "Trend Altı (Ayı) 🔴"
        if son_s['Close'] > son_s['EMA_200']:
            bogalar += 1

        rsi_deger = son_s['RSI_14']
        rsi_durum = "Nötr Seviye ⚪"
        if rsi_deger < 35:
            rsi_durum = "Aşırı Ucuz (Destek Bölgesi) 🟢"
            bogalar += 1
        elif rsi_deger > 70:
            rsi_durum = "Aşırı Şişkin (Direnç Bölgesi) 🔴"
        else:
            bogalar += 0.5

        st_col = next((c for c in df_secilen.columns if c.startswith('SUPERTd')), None)
        is_st_buy = (son_s[st_col] == 1) if st_col else True
        st_durum = "AL Sinyali Aktif 🟢" if is_st_buy else "SAT Sinyali Aktif 🔴"
        if is_st_buy:
            bogalar += 1

        finansal_durum = "Riskli 🔴" if finansal_risk else "Sağlıklı 🟢"
        if not finansal_risk:
            bogalar += 1

        skor = (bogalar / toplam_kriter) * 100
        if skor >= 75:
            karar = "GÜÇLÜ AL (Kademeli Alım)"
        elif skor >= 50:
            karar = "TUT / İZLE (Nötr Pozisyon)"
        else:
            karar = "ZAYIF / KÂR REALİZASYONU"

        momentum_10g = None
        if len(df_secilen) > 10:
            momentum_10g = ((son_s['Close'] / df_secilen['Close'].iloc[-11]) - 1) * 100

        index_df = load_index_data()
        fark = None
        if index_df is not None and len(index_df) > 10 and momentum_10g is not None:
            index_momentum = ((index_df['Close'].iloc[-1] / index_df['Close'].iloc[-11]) - 1) * 100
            fark = momentum_10g - index_momentum

        divergence = detect_rsi_divergence(df_secilen)

        direnc = df_secilen['High'].rolling(60, min_periods=1).max().iloc[-1]
        hedef_fiyat = max(son_s[bbu_col], direnc) if bbu_col else direnc
        stop_loss = son_s['EMA_200']

        veri_baglami = build_veri_baglami(
            secilen_hisse_adi, son_s, ema_durum, rsi_deger, rsi_durum, st_durum, finansal_durum,
            momentum_10g, fark, divergence, hedef_fiyat, stop_loss,
            fk, pddd, cari_oran, net_borc_favok, roe, karar, skor, is_banka
        )

    rapor_key = f"gemini_rapor_{secilen_ticker}"

    if gemini_client is not None and veri_baglami is not None:
        baslat_tiklandi = st.button("🧠 Gemini Analizini Başlat", key=f"rapor_baslat_{secilen_ticker}")
        if baslat_tiklandi:
            rapor_prompt = f"""{GADDAR_PERSONA}

Aşağıdaki güncel piyasa verilerine dayanarak {secilen_hisse_adi} hissesi için TÜRKÇE, uzun ve derinlemesine bir analiz raporu yaz. Rapor TAM OLARAK şu 4 başlığı bu sırayla, aynen bu şekilde (emojili ve iki nokta üst üste ile) kullanmalı; her başlığın altında en az 3-4 cümlelik, veriye dayalı, doğrudan ve sert bir analiz olmalı:

📊 Trend ve İvme Analizi:
🎯 Osilatör ve Güç Kontrolü:
💸 Fintables Temel Analiz Süzgeci:
🚨 Son Karar ve Risk Alarmı:

Kurallar:
- "Fintables Temel Analiz Süzgeci" bölümünde, veri bağlamında Cari Oran veya Net Borç/FAVÖK yoksa bunlardan hiç bahsetme; sadece F/K, PD/DD ve ROE üzerinden yorum yap.
- "Son Karar ve Risk Alarmı" bölümünde asla yuvarlak, muğlak cümle kurma; riskleri ve tuzakları doğrudan söyle.
- Kesin "al/sat" emri verme ama net bir yönelim ve gerekçe sun.
- Her başlığın altında EN FAZLA 2-3 kısa cümle yaz. Uzun, süslü, edebi cümle kurma; bir çalışanın yöneticisine sözlü rapor verir gibi kısa ve net konuş. Sıfat yığma, benzetme yapma, doğrudan olguyu ve sonucu söyle.

Veri Bağlamı:
{veri_baglami}
"""
            with st.spinner("Gemini derinlemesine analiz ediyor..."):
                try:
                    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=rapor_prompt)
                    st.session_state[rapor_key] = response.text
                except Exception as e:
                    st.session_state[rapor_key] = None
                    st.error(f"Gemini isteği başarısız oldu: {e}")

        if st.session_state.get(rapor_key):
            with st.container(border=True):
                st.markdown(st.session_state[rapor_key])
    elif veri_baglami is not None:
        st.info("Gemini derin analiz raporu için proje kök dizinindeki .env dosyasına GEMINI_API_KEY eklemen gerekiyor. Aşağıda temel kural tabanlı özet gösteriliyor.")
        with st.container(border=True):
            st.markdown(
                f"**Strateji:** {karar}  \n"
                f"**Güven Skoru:** %{int(skor)}  \n"
                f"**EMA200 Durumu:** {ema_durum}  \n"
                f"**RSI (14):** {rsi_deger:.2f} ({rsi_durum})  \n"
                f"**SuperTrend:** {st_durum}  \n"
                f"**Finansal Sağlık:** {finansal_durum}"
            )
    else:
        st.warning("Bu hisse için veri yüklenemedi.")

    st.markdown("<div class='fintables-header'>📊 FINTABLES TEMEL ANALİZ ÖZETİ</div>", unsafe_allow_html=True)

    temel_veriler = [
        ("F/K Oranı", fmt(fk)),
        ("Fiyat / Defter Değeri (PD/DD)", fmt(pddd)),
    ]
    if not is_banka:
        temel_veriler.append(("Cari Oran", fmt(cari_oran)))
        temel_veriler.append(("Net Borç / FAVÖK", fmt(net_borc_favok)))
    temel_veriler.append(("Özsermaye Kârlılığı (ROE)", fmt(roe * 100 if roe is not None else None, suffix='%')))

    tablo_satirlari = "".join(f"<tr><td>{anahtar}</td><td>{deger}</td></tr>" for anahtar, deger in temel_veriler)
    st.markdown(f"<table class='fin-table'><tbody>{tablo_satirlari}</tbody></table>", unsafe_allow_html=True)

    if finansal_risk:
        risk_nedenleri = []
        if cari_oran is not None and cari_oran < 1:
            risk_nedenleri.append("Cari Oran 1'in altında (likidite riski)")
        if net_borc_favok is not None and net_borc_favok > 4:
            risk_nedenleri.append("Net Borç/FAVÖK 4'ün üzerinde (borçluluk riski)")

        st.markdown(f"""
        <div class='risk-alarm'>
            <b style='color:#ef4444;'>🚨 RİSK ALARMI</b><br>
            {'; '.join(risk_nedenleri)}.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='fintables-header'>💬 CANLI SORU-CEVAP</div>", unsafe_allow_html=True)

    groq_client = get_groq_client()

    if groq_client is not None and veri_baglami is not None:
        messages_key = f"messages_{secilen_ticker}"

        if messages_key not in st.session_state:
            st.session_state[messages_key] = []

        for msg in st.session_state[messages_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        kullanici_sorusu = st.chat_input("Kanka, bu hissenin teknik görünümünü bir de bana sor...")
        if kullanici_sorusu:
            st.session_state[messages_key].append({"role": "user", "content": kullanici_sorusu})
            with st.chat_message("user"):
                st.markdown(kullanici_sorusu)
            with st.chat_message("assistant"):
                with st.spinner("Analiz ediliyor..."):
                    try:
                        sistem_mesaji = (
                            f"{GADDAR_PERSONA}\n\nGüncel Veri Bağlamı:\n{veri_baglami}\n\n"
                            "Yukarıdaki bağlamda olmayan ortalama hacim, ortalama fiyat, en yüksek/düşük "
                            "seviye veya volatilite gibi hesaplanabilir bir şey sorulursa ASLA tahmin etme; "
                            "hisse_istatistigi_hesapla aracını çağırarak gerçek veriden hesapla. Güncel haber, "
                            "KAP bildirimi, sektör bilgisi veya sitedeki veri setinde hiç olmayan genel bir şey "
                            "sorulursa web_arama_yap aracını çağırarak internetten araştır."
                        )
                        groq_mesajlari = [{"role": "system", "content": sistem_mesaji}] + st.session_state[messages_key]

                        asistan_mesaji = None
                        for _ in range(3):
                            completion = groq_client.chat.completions.create(
                                model=GROQ_MODEL,
                                messages=groq_mesajlari,
                                tools=GROQ_ARAC_TANIMLARI,
                                tool_choice="auto",
                            )
                            asistan_mesaji = completion.choices[0].message
                            if not asistan_mesaji.tool_calls:
                                break

                            groq_mesajlari.append(asistan_mesaji)
                            for tool_call in asistan_mesaji.tool_calls:
                                args = json.loads(tool_call.function.arguments or "{}")
                                if tool_call.function.name == "web_arama_yap":
                                    sonuc = web_arama_yap(args.get("sorgu", ""))
                                else:
                                    sonuc = hisse_istatistigi_hesapla(
                                        df_secilen, args.get("metrik", ""), args.get("gun_sayisi", 30)
                                    )
                                groq_mesajlari.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": sonuc,
                                })

                        cevap_metni = asistan_mesaji.content or "Bir cevap üretemedim."
                        st.markdown(cevap_metni)
                        st.session_state[messages_key].append({"role": "assistant", "content": cevap_metni})
                    except Exception as e:
                        st.error(f"Groq isteği başarısız oldu: {e}")
    else:
        st.info("Canlı soru-cevap için proje kök dizinindeki .env dosyasına GROQ_API_KEY eklemen gerekiyor.")
