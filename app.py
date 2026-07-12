import os
import sqlite3
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots

import data_engine

load_dotenv()


def _get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


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


GEREKLI_INDIKATOR_KOLONLARI = [
    'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
    'EMA_50', 'EMA_200', 'SUPERT_10_3', 'SUPERTd_10_3',
    'DONCHIAN_UST_20', 'DONCHIAN_ALT_20', 'DMP_14', 'DMN_14', 'ADX_14', 'ATR_14',
]


@st.cache_data(ttl=600)
def load_data(ticker):
    df = None
    tablo_olusturuldu = False
    while True:
        conn = sqlite3.connect('bist_portfolio.db')
        try:
            df = pd.read_sql_query(f"SELECT * FROM {ticker}", conn)
            # Eski semadan kalma tablo (indikator kolonlari eksik) - self-heal
            # sadece "tablo yok" hatasinda degil, eksik kolon durumunda da devreye girsin.
            if not tablo_olusturuldu and any(k not in df.columns for k in GEREKLI_INDIKATOR_KOLONLARI):
                conn.close()
                if not _tabloyu_olustur(ticker):
                    return None
                tablo_olusturuldu = True
                continue
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
    /* Durum varyantlari: renk .risk-alarm ile aynı kutuda, sadece kenarlik/arka plan degisiyor.
       Boylece Python tarafinda her seferinde rgba() string'i kurmak yerine tek bir class eklemek yeterli. */
    .risk-alarm.alarm-iyi {
        background: rgba(34, 197, 94, 0.12);
        border-color: #22c55e;
    }
    .risk-alarm.alarm-uyari {
        background: rgba(234, 179, 8, 0.12);
        border-color: #eab308;
    }
    .risk-alarm.alarm-notr {
        background: rgba(255, 255, 255, 0.03);
        border-color: #2a2e39;
        font-weight: 400;
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

# Durum renkleri - uygulama genelinde TEK kaynak (dogrudan hex yazmak yerine bunlar kullanilir)
RENK_IYI = "#22c55e"       # yukselis / hedef / basari
RENK_UYARI = "#eab308"     # karisik / notr-uyari
RENK_KRITIK = "#ef4444"    # dusus / risk / stop-loss ihlali

CHART_BG = "#131722"
GRID_COLOR = "#2a2e39"
TEXT_COLOR = "#d1d4dc"


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


# ============================================================================
# VADE SİNYAL SİSTEMİ — üç vade, üç bağımsız yöntem ailesi:
#   Kısa Vade   : RSI(14) aşırı bölge dönüşü + MACD histogram onayı (momentum)
#   Orta Vade   : Donchian(20) kırılımı + ADX(14)>20 trend-gücü filtresi (breakout)
#   Uzun Vade   : EMA50/EMA200 + SuperTrend yönü (saf trend-takip, karşı trend yok)
# Hepsi data_engine.add_indicators()'in ürettiği ortak kolonlardan okur — ayrı
# veri çekmez/hesaplamaz. Backtest VE "güncel öneri" kartı bu fonksiyonların
# ürettiği AYNI boolean seri/yön bilgisini kullanır (ikinci bir mantık kopyası yok).
# ============================================================================

def _kisa_vade_kosullari(df):
    # RSI 40/60 orta-bant donusu ("asiri" 30/70 esigi + ayni gun MACD sarti
    # birlikte neredeyse hic gerceklesmiyor - iki olayin birkac gun icinde
    # ust uste gelmesini arayarak reaktif ama gercekci bir sinyal uretiliyor).
    rsi = df['RSI_14']
    rsi_onceki = rsi.shift(1)
    macd_hist = df['MACDh_12_26_9']
    rsi_yukari_kesisim = (rsi_onceki < 40) & (rsi >= 40)
    rsi_asagi_kesisim = (rsi_onceki > 60) & (rsi <= 60)
    yakin_zamanda_yukari = rsi_yukari_kesisim.rolling(3, min_periods=1).max().astype(bool)
    yakin_zamanda_asagi = rsi_asagi_kesisim.rolling(3, min_periods=1).max().astype(bool)
    al = yakin_zamanda_yukari & (macd_hist > 0)
    sat = yakin_zamanda_asagi & (macd_hist < 0)
    # Sinyalin ilk gunu sayilsin, ust uste tekrar etmesin
    al = al & ~al.shift(1).fillna(False)
    sat = sat & ~sat.shift(1).fillna(False)
    return al.fillna(False), sat.fillna(False)


def _kisa_vade_gerekce(df, idx, al_mi):
    rsi = df.loc[idx, 'RSI_14']
    macd_hist = df.loc[idx, 'MACDh_12_26_9']
    if al_mi:
        return f"RSI {rsi:.1f} ile 40 bölgesinden yukarı döndü, MACD histogram pozitif ({macd_hist:+.2f})"
    return f"RSI {rsi:.1f} ile 60 bölgesinden aşağı döndü, MACD histogram negatif ({macd_hist:+.2f})"


def _orta_vade_kosullari(df):
    close = df['Close']
    ust = df['DONCHIAN_UST_20']
    alt = df['DONCHIAN_ALT_20']
    adx = df['ADX_14']
    al = (close > ust) & (adx > 20)
    sat = (close < alt) & (adx > 20)
    return al.fillna(False), sat.fillna(False)


def _orta_vade_gerekce(df, idx, al_mi):
    adx = df.loc[idx, 'ADX_14']
    if al_mi:
        seviye = df.loc[idx, 'DONCHIAN_UST_20']
        return f"Fiyat, 20 günlük Donchian üst bandını ({seviye:.2f} TL) kırdı, ADX {adx:.1f} (>20, trend güçlü)"
    seviye = df.loc[idx, 'DONCHIAN_ALT_20']
    return f"Fiyat, 20 günlük Donchian alt bandını ({seviye:.2f} TL) kırdı, ADX {adx:.1f} (>20, trend güçlü)"


def _uzun_vade_yon_serisi(df):
    close = df['Close']
    ema50 = df['EMA_50']
    ema200 = df['EMA_200']
    st_dir = df['SUPERTd_10_3']
    yukselis = (close > ema50) & (ema50 > ema200) & (st_dir == 1)
    dusus = (close < ema50) & (ema50 < ema200) & (st_dir == -1)
    yon = pd.Series('Karışık', index=df.index)
    yon[yukselis.fillna(False)] = 'Yükseliş'
    yon[dusus.fillna(False)] = 'Düşüş'
    return yon


def _uzun_vade_kosullari(df, yon=None):
    if yon is None:
        yon = _uzun_vade_yon_serisi(df)
    yon_onceki = yon.shift(1)
    al = (yon == 'Yükseliş') & (yon_onceki != 'Yükseliş')
    sat = (yon == 'Düşüş') & (yon_onceki != 'Düşüş')
    return al.fillna(False), sat.fillna(False)


def _uzun_vade_gerekce(df, idx, al_mi):
    ema50, ema200 = df.loc[idx, 'EMA_50'], df.loc[idx, 'EMA_200']
    yon_metni = "Yükseliş" if al_mi else "Düşüş"
    return f"EMA50/EMA200 ve SuperTrend {yon_metni} yönünde hizalandı ({ema50:.2f} / {ema200:.2f} TL)"


def _tavsiye_kosullari(kisa_al, kisa_sat, orta_al, orta_sat, uzun_yon):
    # Kapı (gate) mantığı: kısa/orta vadenin ürettiği giriş sinyalleri, SADECE
    # uzun vade (ana trend) o yöndeyse gerçek bir "tavsiye" işlemine dönüşür.
    # Uzun vadenin KENDİ dönüş sinyalini (uzun_al/uzun_sat) kapıya dahil ETMİYORUZ:
    # aksi halde her Yükseliş penceresinin en erken günü hep uzun_al olacağından
    # tek-pozisyon motoru tavsiye'yi uzun vadenin birebir kopyasına indirger.
    # Çıkışta ise temkinli davranılır: kısa/orta SAT sinyali VEYA ana trendin
    # artık Yükseliş'i bırakması (henüz Düşüş'e dönmese bile) pozisyonu kapatır
    # — bu, tavsiye'yi uzun vadeden daha çabuk çıkan, ayrı bir strateji yapar.
    al = (kisa_al | orta_al) & (uzun_yon == 'Yükseliş')
    sat = (kisa_sat | orta_sat) | (uzun_yon != 'Yükseliş')
    return al.fillna(False), sat.fillna(False)


def _tavsiye_gerekce(df, idx, al_mi, kaynaklar):
    yon_metni = "AL" if al_mi else "SAT"
    kaynak_metni = ", ".join(kaynaklar) if kaynaklar else "vade sinyali"
    return f"Uzun vade ana trend {('Yükseliş' if al_mi else 'Düşüş')} yönündeyken {kaynak_metni} {yon_metni} sinyali verdi"


def _islem_listesi_olustur(df, al_kosulu, sat_kosulu, gerekce_al_fn, stop_loss_fn=None):
    """Gerçek bir tek-pozisyon simülasyonu: pozisyon açıkken yeni AL sinyali yeni
    bir işlem SAYMAZ, sadece pozisyon kapalıyken gelen bir AL yeni işlem başlatır.
    stop_loss_fn(df, giris_idx) verilirse, girişteki ATR bazlı stop seviyesi o gün
    itibariyle sabitlenir; günlük Low bu seviyeyi kırarsa pozisyon ters sinyal
    beklenmeden stop-loss'ta kapanır."""
    islemler = []
    pozisyonda = False
    giris_idx = None
    stop_fiyat = None
    for idx in df.index:
        if not pozisyonda:
            if al_kosulu.loc[idx]:
                pozisyonda = True
                giris_idx = idx
                stop_fiyat = stop_loss_fn(df, giris_idx) if stop_loss_fn else None
        else:
            stop_vuruldu = stop_fiyat is not None and df.loc[idx, 'Low'] <= stop_fiyat
            if stop_vuruldu or sat_kosulu.loc[idx]:
                giris_fiyat = df.loc[giris_idx, 'Close']
                cikis_fiyat = stop_fiyat if stop_vuruldu else df.loc[idx, 'Close']
                islemler.append({
                    'giris_idx': giris_idx, 'giris_tarih': df.loc[giris_idx, 'Date'], 'giris_fiyat': giris_fiyat,
                    'cikis_tarih': df.loc[idx, 'Date'], 'cikis_fiyat': cikis_fiyat,
                    'getiri_yuzde': (cikis_fiyat / giris_fiyat - 1) * 100,
                    'acik': False,
                    'cikis_sebep': 'stop-loss' if stop_vuruldu else 'sinyal',
                    'gerekce': gerekce_al_fn(giris_idx),
                })
                pozisyonda = False
                giris_idx = None
                stop_fiyat = None

    if pozisyonda and giris_idx is not None:
        son_idx = df.index[-1]
        giris_fiyat = df.loc[giris_idx, 'Close']
        cikis_fiyat = df.loc[son_idx, 'Close']
        islemler.append({
            'giris_idx': giris_idx, 'giris_tarih': df.loc[giris_idx, 'Date'], 'giris_fiyat': giris_fiyat,
            'cikis_tarih': df.loc[son_idx, 'Date'], 'cikis_fiyat': cikis_fiyat,
            'getiri_yuzde': (cikis_fiyat / giris_fiyat - 1) * 100,
            'acik': True,
            'cikis_sebep': None,
            'gerekce': gerekce_al_fn(giris_idx),
        })
    return islemler


def _istatistik_ozeti(islemler):
    if not islemler:
        return None
    getiriler = [t['getiri_yuzde'] for t in islemler]
    kazanan = sum(1 for g in getiriler if g > 0)
    return {
        'toplam_islem': len(islemler),
        'kazanma_orani': (kazanan / len(islemler)) * 100,
        'ortalama_getiri': sum(getiriler) / len(getiriler),
    }


def _atr_stop_loss_fn(df, giris_idx):
    giris_fiyat = df.loc[giris_idx, 'Close']
    atr = df.loc[giris_idx, 'ATR_14']
    return giris_fiyat - 2 * atr


if not hisseler:
    st.warning("Watchlist'in boş. Yukarıdaki '⚙️ Watchlist Yönetimi' kısmından en az bir hisse ekle.")
    st.stop()

# ============ HESAPLAMA (Günlük Özet için, sadece bir kez) ============

# --- Portföy hesaplama ---
_pf_conn = _watchlist_baglantisi()
portfoy = data_engine.portfoy_getir(_pf_conn)
_pf_conn.close()

pf_satirlari = []
pf_ham_veri = []
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

            pf_satirlari.append({
                'Hisse': ad,
                'Adet': adet,
                'Ort. Maliyet': f"{maliyet:.2f} TL",
                'Güncel Fiyat': f"{guncel_fiyat:.2f} TL",
                'Toplam Değer': f"{pozisyon_degeri:,.2f} TL".replace(',', '.'),
                'Kâr/Zarar': (
                    f"<span style='color:{RENK_IYI if kar_zarar >= 0 else RENK_KRITIK};'>"
                    f"{kar_zarar:+,.2f} TL ({kar_zarar_yuzde:+.1f}%)</span>"
                ).replace(',', '.'),
            })

yogunlasma_uyarilari = _portfoy_risk_uyarilari(pf_ham_veri, toplam_deger)

# ============ GÜNLÜK ÖZET ============
with st.container(border=True):
    st.markdown("#### 🗞️ Günlük Özet")
    if not yogunlasma_uyarilari:
        st.caption("Bugün için özel bir uyarı yok — portföyünde yoğunlaşma riski görünmüyor.")
    else:
        for uyari in yogunlasma_uyarilari:
            st.markdown(
                f"<div class='risk-alarm alarm-uyari'>{uyari}</div>",
                unsafe_allow_html=True
            )

st.markdown("---")

tab_portfoy, tab_teknik = st.tabs(["💼 Portföy", "📈 Teknik Analiz"])

with tab_portfoy:
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
        st.caption(f"ℹ️ Disiplin kuralı: tek pozisyon portföyün %35'ini (≈ {toplam_deger * 0.35:,.2f} TL) geçmemesi önerilir.".replace(',', '.'))

with tab_teknik:
    secilen_hisse_adi = st.selectbox("📊 Detaylı Analiz İçin Hisse Seç:", list(hisseler.values()), key="teknik_hisse_secim")
    secilen_ticker = [k for k, v in hisseler.items() if v == secilen_hisse_adi][0]
    df_secilen = load_data(secilen_ticker)

    if df_secilen is None or df_secilen.empty or len(df_secilen) < 210:
        st.warning("Bu hisse için yeterli veri yok (en az ~210 günlük geçmiş gerekiyor).")
    else:
        son_s = df_secilen.iloc[-1]
        onceki_s = df_secilen.iloc[-2]

        kisa_al_s, kisa_sat_s = _kisa_vade_kosullari(df_secilen)
        orta_al_s, orta_sat_s = _orta_vade_kosullari(df_secilen)
        uzun_yon_s = _uzun_vade_yon_serisi(df_secilen)
        uzun_al_s, uzun_sat_s = _uzun_vade_kosullari(df_secilen, uzun_yon_s)
        tavsiye_al_s, tavsiye_sat_s = _tavsiye_kosullari(kisa_al_s, kisa_sat_s, orta_al_s, orta_sat_s, uzun_yon_s)

        def _guncel_durum(al_serisi, sat_serisi):
            if bool(al_serisi.iloc[-1]):
                return "AL"
            if bool(sat_serisi.iloc[-1]):
                return "SAT"
            return "NÖTR"

        kisa_durum = _guncel_durum(kisa_al_s, kisa_sat_s)
        orta_durum = _guncel_durum(orta_al_s, orta_sat_s)
        uzun_durum = uzun_yon_s.iloc[-1]
        tavsiye_durum = "AL" if bool(tavsiye_al_s.iloc[-1]) else ("SAT" if bool(tavsiye_sat_s.iloc[-1]) else "BEKLE")

        # ============ VADE TABLOSU (öneri kartı) ============
        st.markdown("<div class='fintables-header'>🧭 Vade Tablosu — Güncel Durum</div>", unsafe_allow_html=True)
        vade_renk_harita = {"AL": RENK_IYI, "Yükseliş": RENK_IYI, "SAT": RENK_KRITIK, "Düşüş": RENK_KRITIK,
                             "NÖTR": RENK_UYARI, "Karışık": RENK_UYARI, "BEKLE": RENK_UYARI}
        vade_sinif_harita = {"AL": "alarm-iyi", "Yükseliş": "alarm-iyi", "SAT": "", "Düşüş": "",
                             "NÖTR": "alarm-uyari", "Karışık": "alarm-uyari", "BEKLE": "alarm-uyari"}
        vade_cols = st.columns(4)
        vade_bilgi = [
            ("Kısa Vade", kisa_durum, "RSI dönüşü + MACD"),
            ("Orta Vade", orta_durum, "Donchian kırılımı + ADX"),
            ("Uzun Vade", uzun_durum, "EMA50/200 + SuperTrend"),
            ("Tavsiye", tavsiye_durum, "Uzun vade kapılı bileşik"),
        ]
        for col, (etiket, durum, yontem) in zip(vade_cols, vade_bilgi):
            with col:
                renk = vade_renk_harita[durum]
                sinif = vade_sinif_harita[durum]
                st.markdown(
                    f"<div class='risk-alarm {sinif}' style='text-align:center;'>"
                    f"<b>{etiket}</b><br>"
                    f"<span style='font-size:0.7em; opacity:0.65;'>{yontem}</span><br>"
                    f"<span style='color:{renk}; font-size:1.3em; font-weight:700;'>{durum}</span></div>",
                    unsafe_allow_html=True
                )
        st.caption(
            "ℹ️ Kısa/Orta vade sadece o gün yeni bir sinyal tetiklendiyse AL/SAT gösterir, aksi halde NÖTR. "
            "Uzun vade sürekli bir trend rejimi okumasıdır (Yükseliş/Düşüş/Karışık). Tavsiye, kısa/orta "
            "sinyallerini yalnızca uzun vadenin izin verdiği yönde işleme çevirir."
        )

        # ============ GRAFİK ============
        st.markdown("<div class='fintables-header'>📉 Grafik (Son 1 Yıl)</div>", unsafe_allow_html=True)
        df_gorunum = df_secilen.tail(252).reset_index(drop=True)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.2, 0.25], specs=[[{"secondary_y": True}], [{}], [{}]],
        )

        hacim_renkleri = [
            'rgba(34,197,94,0.35)' if df_gorunum['Close'].iloc[i] >= df_gorunum['Open'].iloc[i] else 'rgba(239,68,68,0.35)'
            for i in range(len(df_gorunum))
        ]
        fig.add_trace(go.Bar(x=df_gorunum['Date'], y=df_gorunum['Volume'], name='Hacim',
                              marker_color=hacim_renkleri, showlegend=False), row=1, col=1, secondary_y=True)
        fig.update_yaxes(range=[0, df_gorunum['Volume'].max() * 4], showticklabels=False, showgrid=False,
                          secondary_y=True, row=1, col=1)

        fig.add_trace(go.Candlestick(
            x=df_gorunum['Date'], open=df_gorunum['Open'], high=df_gorunum['High'],
            low=df_gorunum['Low'], close=df_gorunum['Close'], name='OHLC', showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['EMA_50'], name='EMA 50',
                                  line=dict(color='#f0a500', width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['EMA_200'], name='EMA 200',
                                  line=dict(color='#0891b2', width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['DONCHIAN_UST_20'], name='Donchian',
                                  legendgroup='donchian', line=dict(color='#5b6b8c', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['DONCHIAN_ALT_20'], name='Donchian',
                                  legendgroup='donchian', showlegend=False, line=dict(color='#5b6b8c', width=1),
                                  fill='tonexty', fillcolor='rgba(91,107,140,0.07)'), row=1, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=dict(color=TEXT_COLOR),
            xaxis_rangeslider_visible=False, height=760, margin=dict(l=10, r=10, t=15, b=10), showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                        bgcolor='rgba(0,0,0,0)', font=dict(size=9, color=TEXT_COLOR)),
        )

        hist_renkleri = ['#22c55e' if v >= 0 else '#ef4444' for v in df_gorunum['MACDh_12_26_9'].fillna(0)]
        fig.add_trace(go.Bar(x=df_gorunum['Date'], y=df_gorunum['MACDh_12_26_9'], name='MACD Hist.',
                              marker_color=hist_renkleri, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['MACD_12_26_9'], name='MACD',
                                  line=dict(color='#2962ff', width=1.3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['MACDs_12_26_9'], name='Sinyal',
                                  line=dict(color='#ff9800', width=1.3)), row=2, col=1)

        fig.add_hrect(y0=70, y1=100, fillcolor=RENK_KRITIK, opacity=0.06, line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=RENK_IYI, opacity=0.06, line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['RSI_14'], name='RSI 14',
                                  line=dict(color='#a855f7', width=1.3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_gorunum['Date'], y=df_gorunum['ADX_14'], name='ADX 14',
                                  line=dict(color='#eab308', width=1.1, dash='dot')), row=3, col=1)

        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=1, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=False, row=2, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, showticklabels=True, row=3, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Üst panel: mum + EMA50/EMA200 + Donchian(20) kanalı (hacim arkada yarı saydam). "
            "Orta panel: MACD. Alt panel: RSI(14, gölgeli bölgeler 30/70) + ADX(14, noktalı çizgi, "
            "20 üzeri güçlü trend anlamına gelir)."
        )

        # ============ BACKTEST (4 aşama) ============
        st.markdown("<div class='fintables-header'>🧪 Backtest — 4 Aşama (2 Yıllık Veri)</div>", unsafe_allow_html=True)
        if st.button("Backtest Et", key=f"backtest_btn_{secilen_ticker}"):
            st.session_state[f"bt_{secilen_ticker}"] = True

        if st.session_state.get(f"bt_{secilen_ticker}"):
            baslangic_idx = 210
            df_test = df_secilen.iloc[baslangic_idx:].reset_index(drop=True)

            kisa_al_t, kisa_sat_t = _kisa_vade_kosullari(df_test)
            orta_al_t, orta_sat_t = _orta_vade_kosullari(df_test)
            uzun_yon_t = _uzun_vade_yon_serisi(df_test)
            uzun_al_t, uzun_sat_t = _uzun_vade_kosullari(df_test, uzun_yon_t)
            tavsiye_al_t, tavsiye_sat_t = _tavsiye_kosullari(kisa_al_t, kisa_sat_t, orta_al_t, orta_sat_t, uzun_yon_t)

            al_tut_getiri = (df_test['Close'].iloc[-1] / df_test['Close'].iloc[0] - 1) * 100
            st.caption(
                f"📌 Al-Tut karşılaştırması ({df_test['Date'].iloc[0].strftime('%d.%m.%Y')} — bugün): "
                f"**{al_tut_getiri:+.2f}%** — her aşamanın altındaki sonuçlar bu değerle karşılaştırılmalı."
            )

            asamalar = [
                ("Kısa Vade", kisa_al_t, kisa_sat_t, lambda idx: _kisa_vade_gerekce(df_test, idx, True), '#22c55e', 'triangle-up'),
                ("Orta Vade", orta_al_t, orta_sat_t, lambda idx: _orta_vade_gerekce(df_test, idx, True), '#0284c7', 'diamond'),
                ("Uzun Vade", uzun_al_t, uzun_sat_t, lambda idx: _uzun_vade_gerekce(df_test, idx, True), '#a855f7', 'star'),
                ("Tavsiye", tavsiye_al_t, tavsiye_sat_t, lambda idx: "Kısa/Orta sinyali, uzun vade onaylı", '#f0a500', 'circle'),
            ]

            for ad, al_k, sat_k, gerekce_fn, renk, sembol in asamalar:
                islemler = _islem_listesi_olustur(df_test, al_k, sat_k, gerekce_fn, stop_loss_fn=_atr_stop_loss_fn)
                ozet = _istatistik_ozeti(islemler)
                with st.expander(f"{ad} — {len(islemler)} işlem", expanded=False):
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        st.metric("İşlem Sayısı", ozet['toplam_islem'] if ozet else "—")
                    with stat_cols[1]:
                        st.metric("Kazanma Oranı", f"%{ozet['kazanma_orani']:.0f}" if ozet else "—")
                    with stat_cols[2]:
                        st.metric("Ort. Getiri", f"{ozet['ortalama_getiri']:+.2f}%" if ozet else "—",
                                   delta=f"Al-Tut: {al_tut_getiri:+.1f}%")

                    if islemler:
                        bt_fig = go.Figure()
                        bt_fig.add_trace(go.Candlestick(
                            x=df_test['Date'], open=df_test['Open'], high=df_test['High'],
                            low=df_test['Low'], close=df_test['Close'], name='OHLC', showlegend=False,
                        ))
                        al_df = df_test[al_k]
                        sat_df = df_test[sat_k]
                        if not al_df.empty:
                            bt_fig.add_trace(go.Scatter(x=al_df['Date'], y=al_df['Low'] * 0.97, mode='markers',
                                                         marker=dict(symbol=sembol, size=10, color=renk), name='AL'))
                        if not sat_df.empty:
                            bt_fig.add_trace(go.Scatter(x=sat_df['Date'], y=sat_df['High'] * 1.03, mode='markers',
                                                         marker=dict(symbol=sembol, size=10, color=RENK_KRITIK), name='SAT'))
                        bt_fig.update_layout(
                            template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                            font=dict(color=TEXT_COLOR), xaxis_rangeslider_visible=False, height=320,
                            margin=dict(l=10, r=10, t=15, b=10), showlegend=True,
                            legend=dict(orientation='h', bgcolor='rgba(0,0,0,0)', font=dict(size=9, color=TEXT_COLOR)),
                        )
                        bt_fig.update_xaxes(gridcolor=GRID_COLOR)
                        bt_fig.update_yaxes(gridcolor=GRID_COLOR)
                        st.plotly_chart(bt_fig, use_container_width=True)

                        st.markdown(html_tablo([
                            {
                                'Giriş': f"{t['giris_tarih'].strftime('%d.%m.%Y')} ({t['giris_fiyat']:.2f} TL)",
                                'Çıkış': "Açık" if t['acik'] else f"{t['cikis_tarih'].strftime('%d.%m.%Y')} ({t['cikis_fiyat']:.2f} TL)",
                                'Çıkış Sebebi': "—" if t['acik'] else ("🛑 Stop-Loss (2×ATR)" if t['cikis_sebep'] == 'stop-loss' else "Sinyal"),
                                'Getiri': f"{t['getiri_yuzde']:+.2f}%",
                                'Gerekçe': t['gerekce'],
                            } for t in islemler
                        ]), unsafe_allow_html=True)
                    else:
                        st.caption(f"{ad} bu dönemde hiç işlem üretmedi.")

            st.caption(
                "⚠️ 2 yıllık veriyle sınırlı bir backtest — az sayıda işlem istatistiksel olarak güvenilir "
                "olmayabilir. Stop-loss girişteki 2×ATR(14) seviyesine sabitlenmiştir (trailing değildir). "
                "Geçmiş performans gelecekteki sonuçların garantisi değildir."
            )
