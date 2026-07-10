import json
import os
import re
import sqlite3
import time
from datetime import datetime

import anthropic
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
GROQ_MODEL = "openai/gpt-oss-120b"

_ARAC_SIZINTISI_DESENI = re.compile(r"<function=([\w_]+)>\s*(\{.*?\})\s*</function>", re.DOTALL)


def _sizinti_arac_cagrisini_yakala(metin):
    eslesme = _ARAC_SIZINTISI_DESENI.search(metin or "")
    if not eslesme:
        return None
    try:
        return eslesme.group(1), json.loads(eslesme.group(2))
    except Exception:
        return None

TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")

ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-opus-4-8"


@st.cache_resource
def get_anthropic_client():
    if not ANTHROPIC_API_KEY or "buraya" in ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def claude_ust_akil_raporu(veri_baglami, hisse_adi):
    client = get_anthropic_client()
    if client is None:
        return None, "Claude API anahtarı yapılandırılmamış (ANTHROPIC_API_KEY eksik)."

    prompt = f"""{hisse_adi} hissesi için, aşağıdaki teknik veri özetini başlangıç noktası olarak kullanarak
ama ona bağlı kalmadan, kendi bağımsız araştırmanı (güncel haberler, piyasa koşulları, sektör
durumu) web araması yaparak yürütüp kurumsal kalitede kısa bir yatırım bankası araştırma notu yaz.

Teknik Veri Özeti:
{veri_baglami}

Rapor TAM OLARAK şu formatta olsun:
**Tavsiye:** (AL / TUT / SAT)
**12 Aylık Hedef Fiyat:** (TL)

(3-4 kısa paragraf: güncel piyasa koşulları, sektörel gelişmeler, teknik görünüm özeti — gerçek araştırmana dayanarak)

**Ana Riskler:**
- ...
- ...

En fazla 300-350 kelime yaz, gereksiz uzatma yapma. Türkçe yaz.
En fazla 2 web araması yap; sadece en kritik güncel gelişmeyi ve analist hedef fiyat beklentisini doğrulamak için ara, gereksiz yere fazla arama yapma."""

    try:
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 2}],
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            response = stream.get_final_message()

        if response.stop_reason == "refusal":
            return None, "Claude bu isteği güvenlik politikası gereği reddetti."

        metin_parcalari = [b.text for b in response.content if b.type == "text"]
        rapor = "\n\n".join(metin_parcalari).strip()
        if not rapor:
            return None, "Claude boş bir yanıt döndürdü."
        return rapor, None
    except anthropic.APIStatusError as e:
        return None, f"Claude API hatası: {e.message}"
    except Exception as e:
        return None, f"Claude isteği başarısız oldu: {e}"


ANALIST_PERSONA = (
    "Sen 25 yillik tecrubeye sahip, disiplinli ve dogrudan konusan kidemli bir BIST finans "
    "analistisin. Kullaniciyi yumusak, muglak cumlelerle oyalamazsin; riskleri ve zayifliklari "
    "acik ve net bir dille soylersin. Her iddiani sana verilen sayisal veriye dayandirirsin, "
    "veride olmayan hicbir seyi uydurmazsin. Samimi ama profesyonelsin, kesinlikle Turkce "
    "konusursun ve asla kesin 'al/sat' emri vermezsin, sadece net bir yonelim ve gerekce sunarsin. "
    "Uslubun bir yatirimciya sozlu rapor sunan bir calisan gibi: kisa, duz, sade cumleler "
    "kurarsin. Edebi benzetme, suslu sifat veya 'adeta', 'sanki', 'bir senfoni gibi' turunden "
    "cilali ifadeler kullanmazsin — sadece olguyu ve sonucu soylersin."
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
            "name": "destek_direnc_hesapla",
            "description": (
                "Seçili hissenin GERÇEK fiyat verisinden swing high/low (yerel tepe ve dip) yöntemiyle "
                "destek ve direnç seviyelerini hesaplar. Kullanıcı destek, direnç, kritik seviye, teknik "
                "seviye gibi bir şey sorduğunda ASLA tahmin etme; bu aracı çağırarak gerçek veriden hesapla."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gun_sayisi": {
                        "type": "integer",
                        "description": "Kaç günlük veri üzerinden hesaplanacağı (varsayılan 120)",
                    },
                },
                "required": [],
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


def _seviyeleri_kumele(seviyeler, esik=0.015):
    kumelenmis = []
    for s in seviyeler:
        if kumelenmis and abs(s - kumelenmis[-1]) / kumelenmis[-1] < esik:
            continue
        kumelenmis.append(s)
    return kumelenmis


def _destek_direnc_seviyeleri(df, gun_sayisi=120, guncel_fiyat_override=None):
    gun_sayisi = max(20, min(int(gun_sayisi or 120), len(df)))
    veri = df.tail(gun_sayisi).reset_index(drop=True)
    guncel_fiyat = guncel_fiyat_override if guncel_fiyat_override is not None else veri['Close'].iloc[-1]

    high = veri['High'].values
    low = veri['Low'].values

    tepe_idx = _local_extrema_idx(high, order=3, mode="max")
    dip_idx = _local_extrema_idx(low, order=3, mode="min")

    direnc_seviyeleri = _seviyeleri_kumele(
        sorted(set(round(float(high[i]), 2) for i in tepe_idx if high[i] > guncel_fiyat))
    )[:3]
    destek_seviyeleri = _seviyeleri_kumele(
        sorted(set(round(float(low[i]), 2) for i in dip_idx if low[i] < guncel_fiyat), reverse=True)
    )[:3]

    return guncel_fiyat, direnc_seviyeleri, destek_seviyeleri, gun_sayisi


def destek_direnc_hesapla(df, gun_sayisi=120):
    guncel_fiyat, direnc_seviyeleri, destek_seviyeleri, gun_sayisi_eff = _destek_direnc_seviyeleri(df, gun_sayisi)

    satirlar = [f"Güncel Fiyat: {guncel_fiyat:.2f} TL"]
    satirlar.append(
        "Direnç Seviyeleri (yakından uzağa): " + ", ".join(f"{d:.2f} TL" for d in direnc_seviyeleri)
        if direnc_seviyeleri else "Direnç Seviyeleri: Belirlenemedi (yeterli tepe noktası yok)"
    )
    satirlar.append(
        "Destek Seviyeleri (yakından uzağa): " + ", ".join(f"{d:.2f} TL" for d in destek_seviyeleri)
        if destek_seviyeleri else "Destek Seviyeleri: Belirlenemedi (yeterli dip noktası yok)"
    )
    satirlar.append(f"Analiz Penceresi: Son {gun_sayisi_eff} gün, yerel tepe/dip (swing high/low) yöntemiyle hesaplandı.")
    return "\n".join(satirlar)


def _hacim_teyidi(df, gun=5, referans=20):
    if 'Volume' not in df.columns or len(df) < gun + referans:
        return None
    son_ort = df['Volume'].tail(gun).mean()
    onceki_ort = df['Volume'].iloc[-(gun + referans):-gun].mean()
    if onceki_ort == 0:
        return None
    return son_ort / onceki_ort


def _hacim_teyidi_etiket(oran):
    if oran is None:
        return "Veri Yok"
    if oran > 1.2:
        return f"Güçlü Hacim Teyidi (son {oran:.1f}x)"
    if oran < 0.8:
        return f"Zayıf Hacim ({oran:.1f}x)"
    return f"Normal ({oran:.1f}x)"


def _fibonacci_seviyeleri(df, gun_sayisi=120):
    gun_sayisi = max(20, min(int(gun_sayisi or 120), len(df)))
    veri = df.tail(gun_sayisi).reset_index(drop=True)

    high = veri['High'].values
    low = veri['Low'].values

    tepe_idx = _local_extrema_idx(high, order=3, mode="max")
    dip_idx = _local_extrema_idx(low, order=3, mode="min")

    if tepe_idx and dip_idx:
        en_yakin_tepe_idx = tepe_idx[-1]
        en_yakin_dip_idx = dip_idx[-1]
        tepe = float(high[en_yakin_tepe_idx])
        dip = float(low[en_yakin_dip_idx])
        yon_yukselen = en_yakin_tepe_idx > en_yakin_dip_idx
    else:
        tepe, dip = float(veri['High'].max()), float(veri['Low'].min())
        yon_yukselen = veri['Close'].iloc[-1] >= veri['Close'].iloc[0]

    fark = tepe - dip
    if fark <= 0:
        return None

    oranlar = [0.236, 0.382, 0.5, 0.618, 0.786]
    seviyeler = {o: (tepe - fark * o if yon_yukselen else dip + fark * o) for o in oranlar}
    return {'tepe': tepe, 'dip': dip, 'seviyeler': seviyeler, 'yon': 'yukselen' if yon_yukselen else 'dusen'}


def _vade_ozet(puanlar, gun):
    if not puanlar:
        return {'gun': gun, 'yon': None, 'al_sayisi': 0, 'toplam': 0}
    al_sayisi = sum(1 for p in puanlar if p == 1)
    toplam = len(puanlar)
    if al_sayisi > toplam / 2:
        yon = "Yükseliş"
    elif al_sayisi < toplam / 2:
        yon = "Düşüş"
    else:
        yon = "Karışık"
    return {'gun': gun, 'yon': yon, 'al_sayisi': al_sayisi, 'toplam': toplam}


def _bilesen_ekle(puanlar, bilesenler, yon, ad, deger, veri_tabani):
    puanlar.append(yon)
    bilesenler.append({
        'ad': ad, 'deger': deger, 'yon': 'Yükseliş' if yon == 1 else 'Düşüş',
        'veri_tabani': veri_tabani,
    })


def _zaman_dilimi_analizi(df, son_s, momentum_10g):
    guncel_fiyat = son_s['Close']

    # KISA VADE (10 gün): momentum yönü, StochRSI %K (>50/<50), MACD histogram işareti — hepsi hızlı/reaktif
    kisa_puanlar, kisa_bilesenler = [], []
    if momentum_10g is not None:
        _bilesen_ekle(kisa_puanlar, kisa_bilesenler, 1 if momentum_10g > 0 else -1,
                      '10 Günlük Momentum', f"{momentum_10g:+.2f}%", 'anlık fiyat')
    stoch_k_col = next((c for c in df.columns if c.startswith('STOCHRSIk')), None)
    stoch_k_deger = son_s.get(stoch_k_col) if stoch_k_col else None
    if stoch_k_deger is not None and pd.notna(stoch_k_deger):
        _bilesen_ekle(kisa_puanlar, kisa_bilesenler, 1 if stoch_k_deger > 50 else -1,
                      'StochRSI %K', f"{stoch_k_deger:.1f}", 'son kapanış')
    macd_hist = son_s.get('MACDh_12_26_9')
    if macd_hist is not None and pd.notna(macd_hist):
        _bilesen_ekle(kisa_puanlar, kisa_bilesenler, 1 if macd_hist > 0 else -1,
                      'MACD Histogram', f"{macd_hist:+.3f}", 'son kapanış')
    kisa_vade = _vade_ozet(kisa_puanlar, 10)
    kisa_vade['bilesenler'] = kisa_bilesenler

    # ORTA VADE (50 gün): EMA50-vs-fiyat, 50 günlük momentum, ADX DI+/DI- yönü
    orta_puanlar, orta_bilesenler = [], []
    ema50 = son_s.get('EMA_50')
    if ema50 is not None and pd.notna(ema50):
        _bilesen_ekle(orta_puanlar, orta_bilesenler, 1 if guncel_fiyat > ema50 else -1,
                      'Fiyat / EMA50', f"{guncel_fiyat:.2f} / {ema50:.2f} TL", 'anlık fiyat / son kapanış EMA')
    momentum_50g = None
    if len(df) > 50:
        momentum_50g = ((guncel_fiyat / df['Close'].iloc[-51]) - 1) * 100
        _bilesen_ekle(orta_puanlar, orta_bilesenler, 1 if momentum_50g > 0 else -1,
                      '50 Günlük Momentum', f"{momentum_50g:+.2f}%", 'anlık fiyat')
    di_plus, di_minus = son_s.get('DMP_14'), son_s.get('DMN_14')
    if di_plus is not None and di_minus is not None and pd.notna(di_plus) and pd.notna(di_minus):
        _bilesen_ekle(orta_puanlar, orta_bilesenler, 1 if di_plus > di_minus else -1,
                      'ADX Yönü (+DI/-DI)', f"{di_plus:.1f} / {di_minus:.1f}", 'son kapanış')
    orta_vade = _vade_ozet(orta_puanlar, 50)
    orta_vade['momentum'] = momentum_50g
    orta_vade['bilesenler'] = orta_bilesenler

    # UZUN VADE (200 gün): EMA200-vs-fiyat, SuperTrend yönü, 200 günlük momentum
    uzun_puanlar, uzun_bilesenler = [], []
    ema200 = son_s.get('EMA_200')
    if ema200 is not None and pd.notna(ema200):
        _bilesen_ekle(uzun_puanlar, uzun_bilesenler, 1 if guncel_fiyat > ema200 else -1,
                      'Fiyat / EMA200', f"{guncel_fiyat:.2f} / {ema200:.2f} TL", 'anlık fiyat / son kapanış EMA')
    st_col = next((c for c in df.columns if c.startswith('SUPERTd')), None)
    st_deger = son_s.get(st_col) if st_col else None
    if st_deger is not None and pd.notna(st_deger):
        _bilesen_ekle(uzun_puanlar, uzun_bilesenler, 1 if st_deger == 1 else -1,
                      'SuperTrend Yönü', 'AL' if st_deger == 1 else 'SAT', 'son kapanış')
    momentum_200g = None
    if len(df) > 200:
        momentum_200g = ((guncel_fiyat / df['Close'].iloc[-201]) - 1) * 100
        _bilesen_ekle(uzun_puanlar, uzun_bilesenler, 1 if momentum_200g > 0 else -1,
                      '200 Günlük Momentum', f"{momentum_200g:+.2f}%", 'anlık fiyat')
    uzun_vade = _vade_ozet(uzun_puanlar, 200)
    uzun_vade['momentum'] = momentum_200g
    uzun_vade['bilesenler'] = uzun_bilesenler

    return {'kisa_vade': kisa_vade, 'orta_vade': orta_vade, 'uzun_vade': uzun_vade}


def _vade_celiskisi_kontrolu(zaman_dilimi):
    kisa = zaman_dilimi['kisa_vade']['yon']
    uzun = zaman_dilimi['uzun_vade']['yon']
    if kisa not in ("Yükseliş", "Düşüş") or uzun not in ("Yükseliş", "Düşüş") or kisa == uzun:
        return None
    if kisa == "Düşüş" and uzun == "Yükseliş":
        return ("Kısa vadede (10 gün) zayıflık var ama uzun vadeli (200 gün) trend hâlâ yükseliş yönünde — "
                "bu genel yükseliş trendi içinde kısa vadeli bir düzeltme (pullback) olabilir.")
    return ("Kısa vadede (10 gün) toparlanma var ama uzun vadeli (200 gün) trend hâlâ düşüş yönünde — "
            "bu bir 'tuzak rallisi' (bear market rally) olabilir, temkinli ol.")


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


def _watchlist_baglantisi():
    conn = sqlite3.connect('bist_portfolio.db')
    data_engine.init_ekstra_tablolar(conn)
    return conn


def yeni_hisse_ekle(sembol, ad):
    sembol = sembol.strip().upper()
    if not sembol:
        return False, "Sembol boş olamaz."
    if not sembol.endswith(".IS"):
        sembol += ".IS"
    ticker_db = sembol.replace(".", "_")

    conn = _watchlist_baglantisi()
    try:
        mevcut = data_engine.watchlist_getir(conn)
        if ticker_db in mevcut:
            return False, f"{sembol} zaten watchlist'te."

        with st.spinner(f"{sembol} doğrulanıyor ve ilk verisi çekiliyor..."):
            try:
                taze_df = data_engine.fetch_data(sembol)
            except Exception as e:
                return False, f"{sembol} için veri çekilemedi: {e}"
            if taze_df.empty:
                return False, f"{sembol} geçerli bir hisse gibi görünmüyor (veri boş döndü)."
            taze_df = data_engine.add_indicators(taze_df)
            data_engine.save_to_db(taze_df, sembol, conn)

        data_engine.watchlist_ekle(conn, ticker_db, ad.strip() or sembol.replace(".IS", ""))
        return True, f"{sembol} watchlist'e eklendi."
    finally:
        conn.close()


def hisse_sil(ticker_db):
    conn = _watchlist_baglantisi()
    try:
        data_engine.watchlist_sil(conn, ticker_db)
    finally:
        conn.close()


_wl_conn = _watchlist_baglantisi()
hisseler = data_engine.watchlist_getir(_wl_conn)
_wl_conn.close()


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

    df = df.dropna(subset=['Close']).reset_index(drop=True)
    if df.empty:
        return df

    df['Date'] = pd.to_datetime(df['Date'])

    st_dir_col = next((c for c in df.columns if c.startswith('SUPERTd')), None)
    if st_dir_col:
        df['Sinyal_Degisim'] = df[st_dir_col].diff()

    return df


@st.cache_data(ttl=60)
def get_canli_fiyat(ticker_db):
    """yfinance gunluk mum verisi (history) gecikmeli/eksik olabildigi icin
    (Yahoo bazen o gunun OHLC'sini NaN birakiyor), gercek son fiyati
    fast_info uzerinden ceker. Basarisiz olursa (None, None, None) doner.
    Ucuncu deger, bu fiyatin kontrol edildigi saat (HH:MM) - cache nedeniyle
    en fazla 60 saniyede bir gerceklesir."""
    yf_symbol = ticker_db.replace('_', '.')
    try:
        fi = yf.Ticker(yf_symbol).fast_info
        son_fiyat = fi.get('lastPrice') if hasattr(fi, 'get') else fi.last_price
        onceki_kapanis = fi.get('previousClose') if hasattr(fi, 'get') else fi.previous_close
        if son_fiyat is None or onceki_kapanis is None:
            return None, None, None
        saat = datetime.now().strftime("%H:%M")
        return round(float(son_fiyat), 2), round(float(onceki_kapanis), 2), saat
    except Exception:
        return None, None, None


@st.cache_data(ttl=3600)
def load_index_data():
    try:
        idx = yf.Ticker("XU100.IS").history(period="1y", interval="1d")
        if idx.empty:
            return None
        idx = idx.dropna(subset=['Close'])
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
    .grid-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 4px;
        font-size: 12.5px;
    }
    .grid-table th {
        background-color: #1c2030;
        color: #868c9c;
        font-weight: 700;
        text-align: left;
        padding: 9px 10px;
        border-bottom: 2px solid #2a2e39;
        font-size: 11px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .grid-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #2a2e39;
        color: #d1d4dc;
    }
    .grid-table tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)


def html_tablo(satirlar):
    if not satirlar:
        return ""
    kolonlar = list(satirlar[0].keys())
    basliklar = "".join(f"<th>{k}</th>" for k in kolonlar)
    govde = "".join(
        "<tr>" + "".join(f"<td>{satir[k]}</td>" for k in kolonlar) + "</tr>"
        for satir in satirlar
    )
    return f"<table class='grid-table'><thead><tr>{basliklar}</tr></thead><tbody>{govde}</tbody></table>"

baslik_col, buton_col = st.columns([5, 1])
with baslik_col:
    st.title("📊 BIST Analiz Paneli")
    st.markdown("*Kişisel teknik ve temel analiz aracı*")
with buton_col:
    st.write("")
    st.write("")
    if st.button("🔄 Verileri Yenile", use_container_width=True):
        with st.spinner("Piyasa verileri güncelleniyor..."):
            basarisiz_hisseler = []
            conn = sqlite3.connect(data_engine.DB_PATH)
            try:
                for i, ticker_db in enumerate(hisseler.keys()):
                    ticker = ticker_db.replace('_', '.')
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
        load_data.clear()
        for key in list(st.session_state.keys()):
            if key.startswith(("gemini_", "chat_", "messages_", "claude_")):
                del st.session_state[key]
        st.rerun()

st.markdown("---")

with st.expander("⚙️ Watchlist Yönetimi"):
    ekle_col1, ekle_col2, ekle_col3 = st.columns([2, 2, 1])
    with ekle_col1:
        yeni_sembol = st.text_input("Yeni Hisse Sembolü (örn. SISE.IS)", key="yeni_sembol_input")
    with ekle_col2:
        yeni_ad = st.text_input("Görünen Ad (örn. Şişe Cam)", key="yeni_ad_input")
    with ekle_col3:
        st.write("")
        st.write("")
        if st.button("➕ Ekle", key="hisse_ekle_btn", use_container_width=True):
            basarili, mesaj = yeni_hisse_ekle(yeni_sembol, yeni_ad)
            if basarili:
                st.success(mesaj)
                load_data.clear()
                st.rerun()
            else:
                st.error(mesaj)

    if hisseler:
        st.markdown("**Mevcut Watchlist:**")
        for ticker_db, ad in hisseler.items():
            wl_col1, wl_col2 = st.columns([4, 1])
            with wl_col1:
                st.markdown(f"- {ad} ({ticker_db.replace('_', '.')})")
            with wl_col2:
                if st.button("🗑 Kaldır", key=f"sil_{ticker_db}"):
                    hisse_sil(ticker_db)
                    load_data.clear()
                    st.rerun()

CHART_BG = "#131722"
GRID_COLOR = "#2a2e39"
TEXT_COLOR = "#d1d4dc"


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


def sinyal_performansi(df, al_kosulu, sat_kosulu, son_kac_sinyal=10):
    al_indices = df.index[al_kosulu].tolist()
    sat_indices = df.index[sat_kosulu].tolist()
    if not al_indices:
        return None

    islemler = []
    for giris_idx in al_indices:
        sonraki_satlar = [s for s in sat_indices if s > giris_idx]
        cikis_idx = sonraki_satlar[0] if sonraki_satlar else df.index[-1]
        giris_fiyat = df.loc[giris_idx, 'Close']
        cikis_fiyat = df.loc[cikis_idx, 'Close']
        getiri = (cikis_fiyat / giris_fiyat - 1) * 100
        islemler.append({'getiri': getiri, 'acik': not sonraki_satlar})

    islemler = islemler[-son_kac_sinyal:]
    if not islemler:
        return None

    getiriler = [t['getiri'] for t in islemler]
    kazanan = sum(1 for g in getiriler if g > 0)

    return {
        'toplam_sinyal': len(islemler),
        'kazanma_orani': (kazanan / len(islemler)) * 100,
        'ortalama_getiri': sum(getiriler) / len(getiriler),
        'acik_pozisyon_var': islemler[-1]['acik'],
    }


def _hedef_ve_stop_hesapla(df, son_s):
    guncel_fiyat = son_s['Close']
    _, direnc_seviyeleri, destek_seviyeleri, _ = _destek_direnc_seviyeleri(
        df, gun_sayisi=120, guncel_fiyat_override=guncel_fiyat
    )

    if direnc_seviyeleri:
        hedef_fiyat = direnc_seviyeleri[0]
    else:
        bbu_col = next((c for c in df.columns if c.startswith('BBU_20')), None)
        direnc_60g = df['High'].rolling(60, min_periods=1).max().iloc[-1]
        hedef_fiyat = max(son_s[bbu_col], direnc_60g) if bbu_col else direnc_60g

    stop_loss = destek_seviyeleri[0] if destek_seviyeleri else son_s['EMA_200']
    return hedef_fiyat, stop_loss


def _pozisyon_disiplin_kontrolu(df_pf, guncel_fiyat):
    if df_pf is None or df_pf.empty or 'EMA_200' not in df_pf.columns:
        return None
    hedef_fiyat, stop_loss = _hedef_ve_stop_hesapla(df_pf, df_pf.iloc[-1])
    if guncel_fiyat <= stop_loss:
        return ('stop', stop_loss)
    if guncel_fiyat >= hedef_fiyat:
        return ('hedef', hedef_fiyat)
    return None


def _portfoy_risk_uyarilari(pf_ham_veri, toplam_deger, yogunlasma_esigi=0.35, min_cesitlilik=3):
    uyarilar = []
    if toplam_deger > 0:
        for p in pf_ham_veri:
            oran = p['pozisyon_degeri'] / toplam_deger
            if oran > yogunlasma_esigi:
                uyarilar.append(
                    f"⚠️ {p['ad']}, portföyünün %{oran * 100:.0f}'ini oluşturuyor "
                    f"(önerilen üst sınır: %{yogunlasma_esigi * 100:.0f})."
                )
    if 0 < len(pf_ham_veri) < min_cesitlilik:
        uyarilar.append(
            f"⚠️ Portföyünde sadece {len(pf_ham_veri)} pozisyon var — az sayıda hisseye "
            f"yoğunlaşmak tek bir hissedeki olumsuz haberin toplam etkisini artırır."
        )
    return uyarilar


def build_veri_baglami(hisse_adi, son_s, ema_durum, rsi_deger, rsi_durum, st_durum, finansal_durum,
                        momentum_10g, fark, divergence, hedef_fiyat, stop_loss,
                        fk, pddd, cari_oran, net_borc_favok, roe, karar, skor, is_banka,
                        sinyal_karsilastirma=None, direnc_seviyeleri=None, destek_seviyeleri=None,
                        adx_deger=None, di_plus=None, di_minus=None, hacim_teyidi_oran=None,
                        fibonacci=None, zaman_dilimi=None, portfoy_pozisyonu=None, claude_icin_kisitli=False):
    if portfoy_pozisyonu:
        kar_zarar_tl = f"{portfoy_pozisyonu['kar_zarar']:+,.2f}".replace(',', '.')
        portfoy_satiri = (
            f"{portfoy_pozisyonu['adet']:.0f} adet, {portfoy_pozisyonu['maliyet']:.2f} TL ortalama maliyetle elde "
            f"tutuluyor, güncel kâr/zarar {portfoy_pozisyonu['kar_zarar_yuzde']:+.1f}% ({kar_zarar_tl} TL)"
        )
    else:
        portfoy_satiri = "Bu hissede pozisyon yok"

    satirlar = [
        f"Hisse: {hisse_adi}",
        f"Güncel Fiyat: {son_s['Close']:.2f} TL",
        f"Portföy Pozisyonu: {portfoy_satiri}",
        f"EMA200 Durumu: {ema_durum}",
        f"RSI (14): {rsi_deger:.2f} ({rsi_durum})",
        f"SuperTrend: {st_durum}",
    ]
    if zaman_dilimi:
        for etiket, gun, veri in [
            ("Kısa Vade Trend", 10, zaman_dilimi['kisa_vade']),
            ("Orta Vade Trend", 50, zaman_dilimi['orta_vade']),
            ("Uzun Vade Trend", 200, zaman_dilimi['uzun_vade']),
        ]:
            bilesen_metni = "; ".join(
                f"{b['ad']}: {b['deger']} ({b['yon']}, {b['veri_tabani']})" for b in veri.get('bilesenler', [])
            ) or "Veri Yok"
            satirlar.append(
                f"{etiket} ({gun} gün): {veri['yon'] or 'Belirlenemedi'} "
                f"({veri['al_sayisi']}/{veri['toplam']} ölçüt uyumlu) — {bilesen_metni}"
            )
        satirlar.append(
            "NOT (Vade Analizi Veri Zamanlaması): Momentum ve EMA karşılaştırmaları anlık fiyata göre, "
            "StochRSI/MACD/ADX/SuperTrend değerleri ise son tamamlanmış kapanışa göre hesaplanmıştır — "
            "bu iki tür veri farklı zaman noktalarına ait, karıştırma."
        )
        celiski = _vade_celiskisi_kontrolu(zaman_dilimi)
        if celiski:
            satirlar.append(f"Vade Çelişkisi Uyarısı: {celiski}")
    else:
        satirlar.append("Kısa/Orta/Uzun Vade Trend: Belirlenemedi")
    satirlar += [
        f"10 Günlük Momentum: {fmt(momentum_10g, '%')}",
        f"BIST100'e Göre Göreli Güç Farkı: {fmt(fark, ' puan') if fark is not None else 'Veri Yok'}",
        f"RSI Uyuşmazlığı: {divergence or 'Yok'}",
        f"Direnç Seviyeleri (yakından uzağa, kırılırsa yukarı hareket hızlanır): "
        + (", ".join(f"{d:.2f} TL" for d in direnc_seviyeleri) if direnc_seviyeleri else "Belirlenemedi"),
        f"Destek Seviyeleri (yakından uzağa, kırılırsa aşağı hareket hızlanır): "
        + (", ".join(f"{d:.2f} TL" for d in destek_seviyeleri) if destek_seviyeleri else "Belirlenemedi"),
        f"Hedef Fiyat (En Yakın Direnç Bazlı, yoksa Bollinger/60g üst bant): {hedef_fiyat:.2f} TL",
        f"Stop-Loss (En Yakın Destek Bazlı, yoksa EMA200): {stop_loss:.2f} TL",
        f"F/K Oranı: {fmt(fk)}",
        f"PD/DD: {fmt(pddd)}",
    ]
    if not claude_icin_kisitli:
        satirlar.append(
            f"ADX (Trend Gücü, 14): {fmt(adx_deger)}"
            + (f" (+DI: {fmt(di_plus)}, -DI: {fmt(di_minus)})" if adx_deger is not None else "")
        )
        satirlar.append(f"Hacim Teyidi: {_hacim_teyidi_etiket(hacim_teyidi_oran)}")
        if fibonacci:
            fib_metni = ", ".join(f"%{o*100:.1f}: {v:.2f} TL" for o, v in sorted(fibonacci['seviyeler'].items()))
            satirlar.append(f"Fibonacci Geri Çekilme Seviyeleri ({fibonacci['yon']} hareket, son 120 gün): {fib_metni}")
        else:
            satirlar.append("Fibonacci Geri Çekilme Seviyeleri: Belirlenemedi")
    if not is_banka:
        satirlar.append(f"Cari Oran: {fmt(cari_oran)}")
        satirlar.append(f"Net Borç/FAVÖK: {fmt(net_borc_favok)}")
    satirlar.append(f"Özsermaye Kârlılığı (ROE): {fmt(roe * 100 if roe is not None else None, '%')}")
    satirlar.append(f"Finansal Sağlık: {finansal_durum}")
    satirlar.append(f"Sistemin kural tabanlı kararı: {karar} (Güven Skoru %{int(skor)})")
    if sinyal_karsilastirma:
        satirlar.append("Geçmiş Sinyal Sistemi Performans Karşılaştırması (son sinyaller):")
        for satir in sinyal_karsilastirma:
            satirlar.append(
                f"  * {satir['Sistem']}: {satir['Son Sinyal Sayısı']} sinyal, "
                f"kazanma oranı {satir['Kazanma Oranı']}, ortalama getiri {satir['Ortalama Getiri']}"
                f"{' (son sinyal hâlâ açık pozisyonda)' if satir['Açık Pozisyon'] == 'Evet' else ''}"
            )
    if is_banka:
        satirlar.append("NOT: Bu bir banka hissesidir; Cari Oran ve Net Borç/FAVÖK gibi sanayi rasyoları burada geçersizdir, sadece F/K, PD/DD ve ROE üzerinden bankacılık sağlığını yorumla.")
    return "\n".join(f"- {s}" for s in satirlar)



if not hisseler:
    st.warning("Watchlist'in boş. Yukarıdaki '⚙️ Watchlist Yönetimi' kısmından en az bir hisse ekle.")
    st.stop()

# ============ HESAPLAMA (Günlük Özet ve sekmeler için, sadece bir kez) ============

# --- Toplu Tarama hesaplama ---
tarama_satirlari = []
bugun_sinyal_olanlar = []

with st.spinner("Takip listesi taranıyor..."):
    for ticker_db, ad in hisseler.items():
        df_t = load_data(ticker_db)
        if df_t is None or df_t.empty or len(df_t) < 2:
            continue

        son = df_t.iloc[-1]
        onceki = df_t.iloc[-2]
        canli_fiyat_t, canli_onceki_t, _ = get_canli_fiyat(ticker_db)
        if canli_fiyat_t is not None:
            fiyat_gosterim = canli_fiyat_t
            degisim = ((canli_fiyat_t - canli_onceki_t) / canli_onceki_t) * 100
        else:
            fiyat_gosterim = son['Close']
            degisim = ((son['Close'] - onceki['Close']) / onceki['Close']) * 100

        st_dir_col = next((c for c in df_t.columns if c.startswith('SUPERTd')), None)
        st_al_mi = bool(st_dir_col and son[st_dir_col] == 1)
        st_durum_kisa = "AL 🟢" if st_al_mi else "SAT 🔴"

        sinyal_bugun = []
        if son.get('Sinyal_Degisim') == 2:
            sinyal_bugun.append('SuperTrend AL')
        elif son.get('Sinyal_Degisim') == -2:
            sinyal_bugun.append('SuperTrend SAT')
        if bool(son.get('Optimize_AL', False)):
            sinyal_bugun.append('Optimize AL')
        if bool(son.get('Optimize_SAT', False)):
            sinyal_bugun.append('Optimize SAT')
        if bool(son.get('Squeeze_AL', False)):
            sinyal_bugun.append('Squeeze AL')
        if bool(son.get('Squeeze_SAT', False)):
            sinyal_bugun.append('Squeeze SAT')

        if sinyal_bugun:
            bugun_sinyal_olanlar.append((ad, sinyal_bugun))

        tarama_satirlari.append({
            'Hisse': ad,
            'Fiyat': f"{fiyat_gosterim:.2f} TL",
            'Günlük %': f"{degisim:+.2f}%",
            'RSI': f"{son['RSI_14']:.1f}",
            'SuperTrend': st_durum_kisa,
            'Bugün Sinyal': ', '.join(sinyal_bugun) if sinyal_bugun else '—',
        })

# --- Portföy hesaplama ---
_pf_conn = _watchlist_baglantisi()
portfoy = data_engine.portfoy_getir(_pf_conn)
_pf_conn.close()

pf_satirlari = []
pf_ham_veri = []
pf_disiplin_uyarilari = []
toplam_maliyet = 0.0
toplam_deger = 0.0

if portfoy:
    with st.spinner("Portföy güncelleniyor..."):
        for ticker_db, bilgi in portfoy.items():
            df_pf = load_data(ticker_db)
            if df_pf is None or df_pf.empty:
                continue
            canli_fiyat_pf, _, _ = get_canli_fiyat(ticker_db)
            guncel_fiyat = canli_fiyat_pf if canli_fiyat_pf is not None else df_pf['Close'].iloc[-1]
            ad = hisseler.get(ticker_db, ticker_db)
            adet = bilgi['adet']
            maliyet = bilgi['maliyet']
            pozisyon_maliyeti = adet * maliyet
            pozisyon_degeri = adet * guncel_fiyat
            kar_zarar = pozisyon_degeri - pozisyon_maliyeti
            kar_zarar_yuzde = (kar_zarar / pozisyon_maliyeti * 100) if pozisyon_maliyeti else 0

            toplam_maliyet += pozisyon_maliyeti
            toplam_deger += pozisyon_degeri
            pf_ham_veri.append({'ad': ad, 'pozisyon_degeri': pozisyon_degeri, 'guncel_fiyat': guncel_fiyat, 'ticker_db': ticker_db})

            disiplin = _pozisyon_disiplin_kontrolu(df_pf, guncel_fiyat)
            if disiplin:
                tur, seviye = disiplin
                if tur == 'stop':
                    pf_disiplin_uyarilari.append(('kirmizi', f"🔴 {ad}: fiyat stop-loss seviyesinin ({seviye:.2f} TL) altına indi."))
                else:
                    pf_disiplin_uyarilari.append(('yesil', f"🟢 {ad}: fiyat hedef seviyeye ({seviye:.2f} TL) ulaştı, kâr realizasyonunu değerlendir."))

            pf_satirlari.append({
                'Hisse': ad,
                'Adet': adet,
                'Ort. Maliyet': f"{maliyet:.2f} TL",
                'Güncel Fiyat': f"{guncel_fiyat:.2f} TL",
                'Toplam Değer': f"{pozisyon_degeri:,.2f} TL".replace(',', '.'),
                'Kâr/Zarar': f"{kar_zarar:+,.2f} TL ({kar_zarar_yuzde:+.1f}%)".replace(',', '.'),
            })

yogunlasma_uyarilari = _portfoy_risk_uyarilari(pf_ham_veri, toplam_deger)

# ============ GÜNLÜK ÖZET ============
ozet_maddeleri = []
for ad, sinyaller in bugun_sinyal_olanlar:
    ozet_maddeleri.append(("yesil", f"🔔 {ad}: {', '.join(sinyaller)}"))
for renk, uyari in pf_disiplin_uyarilari:
    ozet_maddeleri.append((renk, uyari))
for uyari in yogunlasma_uyarilari:
    ozet_maddeleri.append(("sari", uyari))

with st.container(border=True):
    st.markdown("#### 🗞️ Günlük Özet")
    if not ozet_maddeleri:
        st.caption("Bugün için özel bir uyarı yok — takip listende yeni sinyal, portföyünde "
                    "stop/hedef ihlali ya da yoğunlaşma riski görünmüyor.")
    else:
        renk_harita = {"yesil": "#22c55e", "kirmizi": "#ef4444", "sari": "#eab308"}
        for renk, metin in ozet_maddeleri:
            st.markdown(
                f"<div class='risk-alarm' style='border-color:{renk_harita[renk]};'>{metin}</div>",
                unsafe_allow_html=True
            )

st.markdown("---")

# ============ SEKMELER ============
tab1, tab2, tab3 = st.tabs(["📊 Genel Bakış", "💼 Portföy", "📈 Teknik Analiz & AI Raporları"])

with tab1:
    with st.spinner("Fiyatlar güncelleniyor..."):
        cols = st.columns(len(hisseler))
        for index, (ticker_db, name) in enumerate(hisseler.items()):
            df = load_data(ticker_db)
            if df is not None and not df.empty and len(df) >= 2:
                son_satir = df.iloc[-1]
                onceki_satir = df.iloc[-2]
                canli_fiyat, canli_onceki_kapanis, canli_saat = get_canli_fiyat(ticker_db)
                if canli_fiyat is not None:
                    fiyat = canli_fiyat
                    yuzde_degisim = round(((fiyat - canli_onceki_kapanis) / canli_onceki_kapanis) * 100, 2)
                else:
                    fiyat = round(son_satir['Close'], 2)
                    yuzde_degisim = round(((fiyat - onceki_satir['Close']) / onceki_satir['Close']) * 100, 2)
                    canli_saat = None

                prefix = "+" if yuzde_degisim > 0 else ""
                saat_etiketi = f" · {canli_saat}" if canli_saat else ""

                with cols[index]:
                    st.metric(
                        label=f"{name} ({ticker_db.replace('_IS', '')}){saat_etiketi}",
                        value=f"{fiyat} TL",
                        delta=f"{prefix}{yuzde_degisim}%"
                    )

    st.markdown("---")
    st.markdown("<div class='fintables-header'>🔍 Toplu Tarama</div>", unsafe_allow_html=True)
    if tarama_satirlari:
        st.markdown(html_tablo(tarama_satirlari), unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='fintables-header'>💼 PORTFÖYÜM</div>", unsafe_allow_html=True)

    with st.expander("➕ Pozisyon Ekle / Güncelle"):
        if hisseler:
            pf_hisse = st.selectbox("Hisse", list(hisseler.values()), key="pf_hisse_secim")
            pf_ticker_db = [k for k, v in hisseler.items() if v == pf_hisse][0]
            mevcut_pozisyon = portfoy.get(pf_ticker_db, {})

            pf_col2, pf_col3, pf_col4 = st.columns([1, 1, 1])
            with pf_col2:
                pf_adet = st.number_input(
                    "Adet", min_value=0.0, step=1.0,
                    value=float(mevcut_pozisyon.get('adet', 0.0)),
                    key=f"pf_adet_input_{pf_ticker_db}"
                )
            with pf_col3:
                pf_maliyet = st.number_input(
                    "Ort. Maliyet (TL)", min_value=0.0, step=0.01,
                    value=float(mevcut_pozisyon.get('maliyet', 0.0)),
                    key=f"pf_maliyet_input_{pf_ticker_db}"
                )
            with pf_col4:
                st.write("")
                st.write("")
                if st.button("💾 Kaydet", key="pf_kaydet_btn", use_container_width=True):
                    _pf_conn2 = _watchlist_baglantisi()
                    if pf_adet <= 0:
                        data_engine.portfoy_sil(_pf_conn2, pf_ticker_db)
                    else:
                        data_engine.portfoy_kaydet(_pf_conn2, pf_ticker_db, pf_adet, pf_maliyet)
                    _pf_conn2.close()
                    st.rerun()

            if mevcut_pozisyon:
                st.caption(f"Mevcut kayıt: {mevcut_pozisyon['adet']:.0f} adet, {mevcut_pozisyon['maliyet']:.2f} TL ortalama maliyet.")
        else:
            st.info("Önce watchlist'e hisse eklemen gerekiyor.")

    if not portfoy:
        st.info("Henüz portföyüne bir pozisyon eklemedin. Yukarıdaki 'Pozisyon Ekle / Güncelle' kısmından başlayabilirsin.")
    elif pf_satirlari:
        st.markdown(html_tablo(pf_satirlari), unsafe_allow_html=True)

        toplam_kz = toplam_deger - toplam_maliyet
        toplam_kz_yuzde = (toplam_kz / toplam_maliyet * 100) if toplam_maliyet else 0
        ozet_cols = st.columns(3)
        with ozet_cols[0]:
            st.metric("Toplam Maliyet", f"{toplam_maliyet:,.2f} TL".replace(',', '.'))
        with ozet_cols[1]:
            st.metric("Güncel Değer", f"{toplam_deger:,.2f} TL".replace(',', '.'))
        with ozet_cols[2]:
            st.metric("Toplam Kâr/Zarar", f"{toplam_kz:,.2f} TL".replace(',', '.'), delta=f"{toplam_kz_yuzde:+.1f}%")

        for uyari in yogunlasma_uyarilari:
            st.markdown(f"<div class='risk-alarm'>{uyari}</div>", unsafe_allow_html=True)
        for renk, uyari in pf_disiplin_uyarilari:
            if renk == 'kirmizi':
                st.markdown(f"<div class='risk-alarm'>{uyari}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='risk-alarm' style='border-color:#22c55e; background: rgba(34,197,94,0.12);'>{uyari}</div>", unsafe_allow_html=True)
        st.caption(f"ℹ️ Disiplin kuralı: tek pozisyon portföyün %35'ini (≈ {toplam_deger * 0.35:,.2f} TL) geçmemesi önerilir.".replace(',', '.'))

with tab3:
    col_left, col_right = st.columns([2.2, 1])

    with col_left:
        secilen_hisse_adi = st.selectbox("📊 Detaylı Analiz İçin Hisse Seç:", list(hisseler.values()))
        secilen_ticker = [k for k, v in hisseler.items() if v == secilen_hisse_adi][0]
        yf_ticker = secilen_ticker.replace('_', '.')

        df_secilen = load_data(secilen_ticker)
        karsilastirma_satirlari = []

        if df_secilen is not None and not df_secilen.empty:
            son_hacim = df_secilen['Volume'].iloc[-1]
            st.markdown(f"**İşlem Hacmi:** {son_hacim:,.0f} LOT".replace(",", "."))

            gun_haritasi = {"1 Ay": 30, "3 Ay": 90, "6 Ay": 180, "1 Yıl": 366}
            zaman_araligi = st.radio(
                "Zaman Aralığı", list(gun_haritasi.keys()), index=3, horizontal=True,
                key=f"zaman_araligi_{secilen_ticker}", label_visibility="collapsed"
            )
            son_tarih = df_secilen['Date'].max()
            baslangic_tarih = son_tarih - pd.Timedelta(days=gun_haritasi[zaman_araligi])
            df_gorunum = df_secilen[df_secilen['Date'] >= baslangic_tarih].reset_index(drop=True)
            if df_gorunum.empty:
                df_gorunum = df_secilen

            bbu_col = next((c for c in df_gorunum.columns if c.startswith('BBU_20')), None)
            bbl_col = next((c for c in df_gorunum.columns if c.startswith('BBL_20')), None)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.025,
                row_heights=[0.5, 0.25, 0.25],
                specs=[[{"secondary_y": True}], [{}], [{}]]
            )

            y1_top = fig.layout.yaxis.domain[1]
            y2_top = fig.layout.yaxis3.domain[1]
            y3_top = fig.layout.yaxis4.domain[1]

            # 0. Hacim (Volume) - fiyat panelinin arkasında yarı saydam
            hacim_renkleri = [
                'rgba(34,197,94,0.35)' if df_gorunum['Close'].iloc[i] >= df_gorunum['Open'].iloc[i] else 'rgba(239,68,68,0.35)'
                for i in range(len(df_gorunum))
            ]
            fig.add_trace(go.Bar(
                x=df_gorunum['Date'], y=df_gorunum['Volume'], name='Hacim',
                marker_color=hacim_renkleri, showlegend=False
            ), row=1, col=1, secondary_y=True)
            max_hacim = df_gorunum['Volume'].max()
            fig.update_yaxes(range=[0, max_hacim * 4], showticklabels=False, showgrid=False, secondary_y=True, row=1, col=1)

            # 1. Candlestick (Mum) Grafik
            fig.add_trace(go.Candlestick(
                x=df_gorunum['Date'],
                open=df_gorunum['Open'],
                high=df_gorunum['High'],
                low=df_gorunum['Low'],
                close=df_gorunum['Close'],
                name='OHLC',
                showlegend=False
            ), row=1, col=1)

            if 'EMA_50' in df_gorunum.columns:
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['EMA_50'], name='EMA 50', legend='legend', line=dict(color='#f0a500', width=1.3)), row=1, col=1)
            if 'EMA_200' in df_gorunum.columns:
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['EMA_200'], name='EMA 200', legend='legend', line=dict(color='#0891b2', width=1.3)), row=1, col=1)

            if bbu_col and bbl_col:
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum[bbu_col], name='Bollinger', legendgroup='bb', legend='legend', line=dict(color='#5b6b8c', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum[bbl_col], name='Bollinger', legendgroup='bb', legend='legend', showlegend=False, line=dict(color='#5b6b8c', width=1), fill='tonexty', fillcolor='rgba(91,107,140,0.08)'), row=1, col=1)

            # SuperTrend (muhafazakar) yön kırılımlarında AL / SAT okları
            if 'Sinyal_Degisim' in df_gorunum.columns:
                al_df = df_gorunum[df_gorunum['Sinyal_Degisim'] == 2]
                sat_df = df_gorunum[df_gorunum['Sinyal_Degisim'] == -2]

                if not al_df.empty:
                    fig.add_trace(go.Scatter(
                        x=al_df['Date'], y=al_df['Low'] * 0.98, mode='markers',
                        marker=dict(symbol='triangle-up', size=11, color='#22c55e'),
                        name='SuperTrend AL', legend='legend'
                    ), row=1, col=1)
                if not sat_df.empty:
                    fig.add_trace(go.Scatter(
                        x=sat_df['Date'], y=sat_df['High'] * 1.02, mode='markers',
                        marker=dict(symbol='triangle-down', size=11, color='#ef4444'),
                        name='SuperTrend SAT', legend='legend'
                    ), row=1, col=1)

            # Optimize Sinyal: MACD kesişimi + SuperTrend trend onaylı, daha sık ama filtreli
            if 'Optimize_AL' in df_gorunum.columns:
                opt_al_df = df_gorunum[df_gorunum['Optimize_AL'].astype(bool)]
                opt_sat_df = df_gorunum[df_gorunum['Optimize_SAT'].astype(bool)]

                if not opt_al_df.empty:
                    fig.add_trace(go.Scatter(
                        x=opt_al_df['Date'], y=opt_al_df['Low'] * 0.995, mode='markers',
                        marker=dict(symbol='diamond', size=8, color='#4ade80', line=dict(width=1, color='#131722')),
                        name='Optimize AL', legend='legend'
                    ), row=1, col=1)
                if not opt_sat_df.empty:
                    fig.add_trace(go.Scatter(
                        x=opt_sat_df['Date'], y=opt_sat_df['High'] * 1.005, mode='markers',
                        marker=dict(symbol='diamond', size=8, color='#f87171', line=dict(width=1, color='#131722')),
                        name='Optimize SAT', legend='legend'
                    ), row=1, col=1)

            # Squeeze Breakout: Bollinger sıkışması sonrası kırılım
            if 'Squeeze_AL' in df_gorunum.columns:
                sq_al_df = df_gorunum[df_gorunum['Squeeze_AL'].astype(bool)]
                sq_sat_df = df_gorunum[df_gorunum['Squeeze_SAT'].astype(bool)]

                if not sq_al_df.empty:
                    fig.add_trace(go.Scatter(
                        x=sq_al_df['Date'], y=sq_al_df['Low'] * 0.97, mode='markers',
                        marker=dict(symbol='star', size=12, color='#22d3ee', line=dict(width=1, color='#131722')),
                        name='Squeeze AL', legend='legend'
                    ), row=1, col=1)
                if not sq_sat_df.empty:
                    fig.add_trace(go.Scatter(
                        x=sq_sat_df['Date'], y=sq_sat_df['High'] * 1.03, mode='markers',
                        marker=dict(symbol='star', size=12, color='#fb923c', line=dict(width=1, color='#131722')),
                        name='Squeeze SAT', legend='legend'
                    ), row=1, col=1)

            # 2. MACD Paneli
            if 'MACD_12_26_9' in df_gorunum.columns:
                hist_renkleri = ['#22c55e' if v >= 0 else '#ef4444' for v in df_gorunum['MACDh_12_26_9'].fillna(0)]
                fig.add_trace(go.Bar(x=df_gorunum['Date'], y=df_gorunum['MACDh_12_26_9'], name='Histogram', marker_color=hist_renkleri, showlegend=False), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['MACD_12_26_9'], name='MACD', legend='legend2', line=dict(color='#2962ff', width=1.3)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['MACDs_12_26_9'], name='Sinyal', legend='legend2', line=dict(color='#ff9800', width=1.3)), row=2, col=1)

            # 3. RSI ve Stokastik RSI Paneli
            if 'RSI_14' in df_gorunum.columns:
                fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['RSI_14'], name='RSI 14', legend='legend3', line=dict(color='#a855f7', width=1.3)), row=3, col=1)
                fig.add_shape(type="line", x0=df_gorunum['Date'].min(), y0=70, x1=df_gorunum['Date'].max(), y1=70, line=dict(color="#ef4444", dash="dash", width=1), row=3, col=1)
                fig.add_shape(type="line", x0=df_gorunum['Date'].min(), y0=30, x1=df_gorunum['Date'].max(), y1=30, line=dict(color="#22c55e", dash="dash", width=1), row=3, col=1)

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
            )
            fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=1, col=1)
            fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=2, col=1)
            fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=True, row=3, col=1)
            fig.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig, use_container_width=True)

            sinyal_sistemleri = [
                ('SuperTrend', df_secilen.get('Sinyal_Degisim') == 2, df_secilen.get('Sinyal_Degisim') == -2),
                ('Optimize', df_secilen.get('Optimize_AL', pd.Series(False, index=df_secilen.index)).astype(bool),
                 df_secilen.get('Optimize_SAT', pd.Series(False, index=df_secilen.index)).astype(bool)),
                ('Squeeze Breakout', df_secilen.get('Squeeze_AL', pd.Series(False, index=df_secilen.index)).astype(bool),
                 df_secilen.get('Squeeze_SAT', pd.Series(False, index=df_secilen.index)).astype(bool)),
            ]

            karsilastirma_satirlari = []
            for sistem_adi, al_kosulu, sat_kosulu in sinyal_sistemleri:
                perf = sinyal_performansi(df_secilen, al_kosulu, sat_kosulu)
                if perf:
                    karsilastirma_satirlari.append({
                        'Sistem': sistem_adi,
                        'Son Sinyal Sayısı': perf['toplam_sinyal'],
                        'Kazanma Oranı': f"%{perf['kazanma_orani']:.0f}",
                        'Ortalama Getiri': f"%{perf['ortalama_getiri']:+.2f}",
                        'Açık Pozisyon': "Evet" if perf['acik_pozisyon_var'] else "Hayır",
                    })

            if karsilastirma_satirlari:
                st.markdown("<div class='fintables-header'>📈 Sinyal Performans Karşılaştırması (Geçmiş)</div>", unsafe_allow_html=True)
                st.markdown(html_tablo(karsilastirma_satirlari), unsafe_allow_html=True)
                st.caption("⚠️ Geçmiş sinyal performansı gelecekteki sonuçların garantisi değildir.")

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
        veri_baglami_claude = None
        karar = skor = ema_durum = rsi_durum = st_durum = finansal_durum = None
        rsi_deger = None

        st.markdown("<div class='fintables-header'>🧠 GEMİNİ DERİN ANALİZ RAPORU</div>", unsafe_allow_html=True)

        if df_secilen is not None and not df_secilen.empty:
            son_s = df_secilen.iloc[-1].copy()
            canli_fiyat_secilen, _, canli_saat_secilen = get_canli_fiyat(secilen_ticker)
            if canli_fiyat_secilen is not None:
                son_s['Close'] = canli_fiyat_secilen
                st.caption(f"Güncel Fiyat: {son_s['Close']:.2f} TL · {canli_saat_secilen}")

            bogalar = 0
            toplam_kriter = 5

            ema_durum = "Trend Üstü (Boğa) 🟢" if son_s['Close'] > son_s['EMA_200'] else "Trend Altı (Ayı) 🔴"
            if son_s['Close'] > son_s['EMA_200']:
                bogalar += 1

            rsi_deger = son_s['RSI_14']
            rsi_durum = "Nötr Seviye ⚪"
            if rsi_deger < 30:
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

            hedef_fiyat, stop_loss = _hedef_ve_stop_hesapla(df_secilen, son_s)

            _, direnc_seviyeleri, destek_seviyeleri, _ = _destek_direnc_seviyeleri(
                df_secilen, gun_sayisi=120, guncel_fiyat_override=son_s['Close']
            )

            adx_deger = son_s.get('ADX_14')
            di_plus = son_s.get('DMP_14')
            di_minus = son_s.get('DMN_14')
            hacim_teyidi_oran = _hacim_teyidi(df_secilen)
            fibonacci = _fibonacci_seviyeleri(df_secilen)
            zaman_dilimi = _zaman_dilimi_analizi(df_secilen, son_s, momentum_10g)

            portfoy_pozisyonu = None
            mevcut_pf_kaydi = portfoy.get(secilen_ticker)
            if mevcut_pf_kaydi:
                pf_adet_sec = mevcut_pf_kaydi['adet']
                pf_maliyet_sec = mevcut_pf_kaydi['maliyet']
                pf_pozisyon_maliyeti_sec = pf_adet_sec * pf_maliyet_sec
                pf_pozisyon_degeri_sec = pf_adet_sec * son_s['Close']
                pf_kar_zarar_sec = pf_pozisyon_degeri_sec - pf_pozisyon_maliyeti_sec
                portfoy_pozisyonu = {
                    'adet': pf_adet_sec,
                    'maliyet': pf_maliyet_sec,
                    'kar_zarar': pf_kar_zarar_sec,
                    'kar_zarar_yuzde': (pf_kar_zarar_sec / pf_pozisyon_maliyeti_sec * 100) if pf_pozisyon_maliyeti_sec else 0,
                }

            veri_baglami = build_veri_baglami(
                secilen_hisse_adi, son_s, ema_durum, rsi_deger, rsi_durum, st_durum, finansal_durum,
                momentum_10g, fark, divergence, hedef_fiyat, stop_loss,
                fk, pddd, cari_oran, net_borc_favok, roe, karar, skor, is_banka,
                karsilastirma_satirlari, direnc_seviyeleri, destek_seviyeleri,
                adx_deger, di_plus, di_minus, hacim_teyidi_oran, fibonacci, zaman_dilimi,
                portfoy_pozisyonu
            )
            veri_baglami_claude = build_veri_baglami(
                secilen_hisse_adi, son_s, ema_durum, rsi_deger, rsi_durum, st_durum, finansal_durum,
                momentum_10g, fark, divergence, hedef_fiyat, stop_loss,
                fk, pddd, cari_oran, net_borc_favok, roe, karar, skor, is_banka,
                karsilastirma_satirlari, direnc_seviyeleri, destek_seviyeleri,
                zaman_dilimi=zaman_dilimi, portfoy_pozisyonu=portfoy_pozisyonu, claude_icin_kisitli=True
            )

            if zaman_dilimi:
                vade_cols = st.columns(3)
                vade_bilgi = [
                    ("Kısa Vade", zaman_dilimi['kisa_vade']),
                    ("Orta Vade", zaman_dilimi['orta_vade']),
                    ("Uzun Vade", zaman_dilimi['uzun_vade']),
                ]
                for col, (etiket, veri) in zip(vade_cols, vade_bilgi):
                    with col:
                        bilesen_satirlari = ""
                        for b in veri.get('bilesenler', []):
                            nokta_rengi = "#22c55e" if b['yon'] == "Yükseliş" else "#ef4444"
                            bilesen_satirlari += (
                                f"<div style='text-align:left; font-size:0.78em; margin-top:4px;'>"
                                f"<span style='color:{nokta_rengi};'>●</span> "
                                f"{b['ad']}: <b>{b['deger']}</b> <span style='opacity:0.6;'>({b['veri_tabani']})</span></div>"
                            )
                        if veri['yon']:
                            renk = "#22c55e" if veri['yon'] == "Yükseliş" else ("#ef4444" if veri['yon'] == "Düşüş" else "#eab308")
                            st.markdown(
                                f"<div class='risk-alarm' style='border-color:{renk}; background: rgba(0,0,0,0.15);'>"
                                f"<div style='text-align:center;'><b>{etiket} ({veri['gun']} gün)</b><br>"
                                f"<span style='color:{renk}; font-size:1.15em;'>{veri['yon']}</span><br>"
                                f"<span style='font-size:0.8em;'>{veri['al_sayisi']}/{veri['toplam']} ölçüt uyumlu</span></div>"
                                f"{bilesen_satirlari}</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div class='risk-alarm' style='text-align:center;'><b>{etiket} ({veri['gun']} gün)</b><br>Belirlenemedi</div>",
                                unsafe_allow_html=True
                            )
                st.caption(
                    "ℹ️ Vade analizindeki momentum/EMA karşılaştırmaları anlık fiyata göre, StochRSI/MACD/ADX/SuperTrend "
                    "ise son tamamlanmış kapanışa göre hesaplanır — parantez içi etiketler hangisinin hangi olduğunu gösterir. "
                    "Bu analiz, ADX, hacim teyidi, destek/direnç ve Fibonacci seviyeleri gibi destekleyici bağlamdır — "
                    "geçmiş veriyle test edilmemiştir. Sadece aşağıdaki 'Sinyal Performans Karşılaştırması' tablosundaki "
                    "SuperTrend/Optimize/Squeeze sinyalleri geçmiş veriyle test edilmiştir."
                )
                celiski = _vade_celiskisi_kontrolu(zaman_dilimi)
                if celiski:
                    st.markdown(
                        f"<div class='risk-alarm' style='border-color:#eab308; background: rgba(234,179,8,0.12);'>"
                        f"⚠️ <b>Vade Çelişkisi:</b> {celiski}</div>",
                        unsafe_allow_html=True
                    )

        with st.expander("🏛️ Bağımsız Araştırma Raporu (Claude)", expanded=False):
            claude_client = get_anthropic_client()
            ust_akil_key = f"claude_rapor_{secilen_ticker}"
            if claude_client is None:
                st.info("Bu özellik için proje kök dizinindeki .env dosyasına ANTHROPIC_API_KEY eklemen gerekiyor.")
            elif veri_baglami_claude is not None:
                if st.button("🔍 Claude ile Bağımsız Araştırma Yap", key=f"claude_btn_{secilen_ticker}"):
                    with st.spinner("Claude web'de araştırma yapıyor ve rapor hazırlıyor... (bir dakika kadar sürebilir)"):
                        claude_rapor, claude_hata = claude_ust_akil_raporu(veri_baglami_claude, secilen_hisse_adi)
                        if claude_hata:
                            st.error(claude_hata)
                        else:
                            st.session_state[ust_akil_key] = claude_rapor
                if st.session_state.get(ust_akil_key):
                    with st.container(border=True):
                        st.markdown(st.session_state[ust_akil_key])
                    st.caption("⚠️ Bu rapor yapay zeka tarafından üretilmiştir, yatırım tavsiyesi değildir.")
            else:
                st.info("Bu hisse için veri yüklenemedi.")

        rapor_key = f"gemini_rapor_{secilen_ticker}"

        if gemini_client is not None and veri_baglami is not None:
            baslat_tiklandi = st.button("🧠 Gemini Analizini Başlat", key=f"rapor_baslat_{secilen_ticker}")
            if baslat_tiklandi:
                rapor_prompt = f"""{ANALIST_PERSONA}

    Aşağıdaki güncel piyasa verilerine dayanarak {secilen_hisse_adi} hissesi için TÜRKÇE, uzun ve derinlemesine bir analiz raporu yaz. Rapor TAM OLARAK şu 5 başlığı bu sırayla, aynen bu şekilde (emojili ve iki nokta üst üste ile) kullanmalı; her başlığın altında en az 3-4 cümlelik, veriye dayalı, doğrudan ve sert bir analiz olmalı:

    📊 Trend ve İvme Analizi:
    🎯 Osilatör ve Güç Kontrolü:
    💸 Temel Analiz ve Değerleme Özeti:
    🗺️ Yol Haritası (Destek/Direnç ve Kırılım Noktaları):
    🚨 Son Karar ve Risk Alarmı:

    Kurallar:
    - "Temel Analiz ve Değerleme Özeti" bölümünde, veri bağlamında Cari Oran veya Net Borç/FAVÖK yoksa bunlardan hiç bahsetme; sadece F/K, PD/DD ve ROE üzerinden yorum yap.
    - "Yol Haritası" bölümünde veri bağlamındaki Direnç ve Destek Seviyelerini birebir kullan: en yakın direncin üzerinde kırılım olursa fiyatın hangi seviyeye (bir sonraki dirence) doğru hareket edebileceğini, en yakın desteğin altında kırılım olursa hangi seviyeye kadar sarkabileceğini somut TL rakamlarıyla söyle. Uydurma seviye kullanma, sadece verilenleri kullan.
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

        st.markdown("<div class='fintables-header'>📊 TEMEL ANALİZ ÖZETİ</div>", unsafe_allow_html=True)

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

            sohbet_kutusu = st.container(height=420, border=True)

            with sohbet_kutusu:
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
                                    f"{ANALIST_PERSONA}\n\nGüncel Veri Bağlamı:\n{veri_baglami}\n\n"
                                    "Yukarıdaki bağlamda olmayan ortalama hacim, ortalama fiyat, en yüksek/düşük "
                                    "seviye veya volatilite gibi hesaplanabilir bir şey sorulursa ASLA tahmin etme; "
                                    "hisse_istatistigi_hesapla aracını çağırarak gerçek veriden hesapla. Destek, direnç "
                                    "veya kritik teknik seviye sorulursa destek_direnc_hesapla aracını çağır. Güncel haber, "
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

                                    if asistan_mesaji.tool_calls:
                                        groq_mesajlari.append(asistan_mesaji)
                                        for tool_call in asistan_mesaji.tool_calls:
                                            args = json.loads(tool_call.function.arguments or "{}")
                                            if tool_call.function.name == "web_arama_yap":
                                                sonuc = web_arama_yap(args.get("sorgu", ""))
                                            elif tool_call.function.name == "destek_direnc_hesapla":
                                                sonuc = destek_direnc_hesapla(df_secilen, args.get("gun_sayisi", 120))
                                            else:
                                                sonuc = hisse_istatistigi_hesapla(
                                                    df_secilen, args.get("metrik", ""), args.get("gun_sayisi", 30)
                                                )
                                            groq_mesajlari.append({
                                                "role": "tool",
                                                "tool_call_id": tool_call.id,
                                                "content": sonuc,
                                            })
                                        continue

                                    # Bazı modeller ara sıra gerçek tool_calls yerine fonksiyon çağrısını
                                    # düz metin olarak sızdırıyor (<function=...>...</function>). Bunu
                                    # yakalayıp kullanıcıya çiğ metin göstermeden aracı biz çalıştırıyoruz.
                                    sizinti = _sizinti_arac_cagrisini_yakala(asistan_mesaji.content)
                                    if sizinti:
                                        arac_adi, args = sizinti
                                        if arac_adi == "web_arama_yap":
                                            sonuc = web_arama_yap(args.get("sorgu", ""))
                                        elif arac_adi == "destek_direnc_hesapla":
                                            sonuc = destek_direnc_hesapla(df_secilen, args.get("gun_sayisi", 120))
                                        else:
                                            sonuc = hisse_istatistigi_hesapla(
                                                df_secilen, args.get("metrik", ""), args.get("gun_sayisi", 30)
                                            )
                                        groq_mesajlari.append({"role": "assistant", "content": asistan_mesaji.content})
                                        groq_mesajlari.append({
                                            "role": "user",
                                            "content": (
                                                f"Araç sonucu: {sonuc}\n\nBuna dayanarak soruyu normal, düz metinle "
                                                "cevapla; fonksiyon çağrısı söz dizimi (<function=...>) kullanma."
                                            ),
                                        })
                                        continue

                                    break

                                cevap_metni = asistan_mesaji.content or "Bir cevap üretemedim."
                                st.markdown(cevap_metni)
                                st.session_state[messages_key].append({"role": "assistant", "content": cevap_metni})
                            except Exception as e:
                                st.error(f"Groq isteği başarısız oldu: {e}")
        else:
            st.info("Canlı soru-cevap için proje kök dizinindeki .env dosyasına GROQ_API_KEY eklemen gerekiyor.")
