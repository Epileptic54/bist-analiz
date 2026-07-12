import os
import sqlite3
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

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
RENK_KRITIK = "#ef4444"    # dusus / risk / stop-loss ihlali


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
