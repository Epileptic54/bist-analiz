# BIST Dashboard Projesi Kuralları

## Proje Amacı
Kişisel bir BIST (Borsa İstanbul) teknik/temel analiz ve karar destek panelim. Watchlist dinamik (SQLite'ta saklanıyor, arayüzden ekle/çıkar) — sabit bir hisse listesi yok, varsayılan olarak ASELS/ASTOR/THYAO/BIMAS/AKBNK ile başladı ama kullanıcı istediği hisseyi ekleyebiliyor. Portföy takibi (adet/maliyet/kâr-zarar) da dahil.

## Teknik Yığın
- Backend/veri: Python, `yfinance` (ücretsiz, resmi olmayan kaynak — bilinen kırılganlıkları var, aşağıya bakın).
- Arayüz: Streamlit, `app.py` tek dosya (~1650 satır) + `data_engine.py` (veri çekme/indikatör hesaplama).
- Depolama: yerel SQLite (`bist_portfolio.db`) — ticker başına tablo (`ASELS_IS` vb.) + `watchlist`/`portfoy` tabloları. **Streamlit Cloud'da bu disk geçici (ephemeral)** — redeploy/restart'ta veri kaybolabilir, henüz çözülmedi.
- Grafik: Plotly, 3 panelli (mum+EMA+Bollinger+hacim+sinyal işaretleri / MACD / RSI).

## İndikatörler (tamamı pure pandas — `pandas_ta`/`numba` KULLANILMIYOR, Streamlit Cloud'un Python sürümüyle uyumsuz çıktığı için tüm indikatörler `data_engine.py`'de elle yazıldı)
EMA(50,200), RSI(14), SuperTrend(10,3), Stokastik RSI, MACD(12,26,9), Bollinger Bands+Bandwidth, ADX/DI+/DI-(14), OBV. Ayrıca `app.py`'de (DB'ye yazılmayan, anlık hesaplanan): destek/direnç seviyeleri (swing high/low), Fibonacci geri çekilme (aynı swing noktalarına dayanıyor), hacim teyidi oranı.

## Kısa/Orta/Uzun Vade Trend Sistemi
Tek bir "trend okuması" yerine göstergeler **gerçek vadelerine göre** gruplanıyor:
- **Kısa Vade (10 gün):** 10 günlük momentum, StochRSI %K, MACD histogram
- **Orta Vade (50 gün):** EMA50 durumu, 50 günlük momentum, ADX yönü
- **Uzun Vade (200 gün):** EMA200 durumu, SuperTrend yönü, 200 günlük momentum

Her vade kartı hangi ölçütün ne dediğini (değer + yön + hangi veriye dayandığı: anlık fiyat mı, son kapanış mı) açıkça gösteriyor. Kısa/uzun vade ters yöne çıkarsa "vade çelişkisi" uyarısı tetikleniyor (düzeltme mi, tuzak rallisi mi).

## Sinyal Sistemleri (backtest edilmiş)
SuperTrend, Optimize (MACD kesişimi + SuperTrend onaylı), Squeeze Breakout (Bollinger sıkışması sonrası kırılım) — `sinyal_performansi()` ile gerçek geçmiş kazanma oranı/ortalama getiri hesaplanıyor. **ADX/hacim teyidi/destek-direnç/Fibonacci/vade analizi backtest EDİLMEMİŞTİR** — arayüzde bu ayrım açıkça belirtiliyor.

## Risk Yönetimi Katmanı
Portföyde pozisyon başına stop-loss/hedef fiyat disiplin kontrolü (en yakın destek/direnç bazlı), yoğunlaşma uyarısı (tek pozisyon %35'i geçerse), çeşitlilik uyarısı (<3 pozisyon). "Günlük Özet" kutusu bunları + bugünkü sinyalleri sekmelerin üstünde tek yerde topluyor.

## Üç AI Entegrasyonu (mimarisi sabit — değiştirilmeden önce kullanıcıya sorulmalı)
- **Gemini** (`gemini-2.5-flash-lite`): buton tetiklemeli derin analiz raporu, 5 başlıklı format.
- **Claude** (`claude-opus-4-8`, Anthropic API): nadir tetiklenen bağımsız araştırma raporu, native web search. **Maliyet kontrolü**: `build_veri_baglami(..., claude_icin_kisitli=True)` ile ADX/hacim teyidi/Fibonacci detayları Claude'a gitmiyor (input token maliyetini şişirmesin diye) — ama Trend Okuması, destek/direnç, portföy pozisyonu gibi merkezi bilgiler HER İKİ varyanta da gidiyor.
- **Groq** (`openai/gpt-oss-120b`): canlı sohbet, tool-calling (`hisse_istatistigi_hesapla`, `destek_direnc_hesapla`, `web_arama_yap` via Tavily).

Üçü de `build_veri_baglami()` fonksiyonundan aynı bağlamı okuyor — yeni bir veri eklerken bu fonksiyona işlemek, otomatik olarak üçüne de yayılmasını sağlıyor.

## Canlı Fiyat
Günlük mum verisi (`history()`) bazen o günün OHLC'sini NaN bırakıyor veya bir gün gecikiyor — bu yüzden `get_canli_fiyat()` `fast_info` üzerinden anlık fiyatı ayrıca çekiyor (60sn cache), başarısız olursa son kapanışa düşüyor. Fiyatın yanında kontrol saati (HH:MM) gösteriliyor. **Not:** momentum/EMA karşılaştırmaları anlık fiyatı kullanıyor ama StochRSI/MACD/ADX/SuperTrend son tamamlanmış kapanışa göre (günlük veri olduğu için tam canlı hesaplanamıyor) — bu her vade bileşeninin yanında açıkça etiketleniyor.

## Arayüz Yapısı
3 sekme: 📊 Genel Bakış (üst metrikler + toplu tarama), 💼 Portföy, 📈 Teknik Analiz & AI Raporları. Sekmelerin üstünde her zaman görünen Günlük Özet kutusu. Watchlist Yönetimi sekmelerin üstünde (her sekmeden erişilebilir).

## Doğrulama Alışkanlığı (bu proje için önemli)
- `python -m py_compile app.py data_engine.py` her değişiklikten sonra.
- Saf fonksiyonları gerçek ASELS/AKBNK verisiyle scratchpad'de izole test et (app.py'yi doğrudan import etme — üst seviyede DB bağlantısı/`st.stop()` çalıştırıyor).
- **Uçtan uca doğrulama için `streamlit.testing.v1.AppTest` kullan** — düz bir `streamlit run` + HTTP GET script'i gerçekten ÇALIŞTIRMAZ (bu oturumda öğrenildi), sadece statik sayfa döner. `AppTest.from_file(...).run()` + `at.exception` kontrolü tek güvenilir yöntem.
- DB şemasına yeni kolon eklerken (yeni indikatör vb.) `load_data()`'nın self-heal mekanizması sadece "tablo yok" hatasında devreye giriyor, mevcut tabloya otomatik kolon eklemiyor — eski ticker tablolarını `DROP TABLE` ile silip yeniden oluşturmak gerekiyor (watchlist/portfoy tablolarına dokunmadan).
