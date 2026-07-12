# BIST Dashboard Projesi Kuralları

## Proje Amacı
Kişisel bir BIST (Borsa İstanbul) portföy takip panelim. Watchlist dinamik (SQLite'ta saklanıyor, arayüzden ekle/çıkar) — sabit bir hisse listesi yok, varsayılan olarak ASELS/ASTOR/THYAO/BIMAS/AKBNK ile başladı ama kullanıcı istediği hisseyi ekleyebiliyor. Portföy takibi (adet/maliyet/kâr-zarar) ana özellik.

## Teknik Yığın
- Backend/veri: Python, `yfinance` (ücretsiz, resmi olmayan kaynak — bilinen kırılganlıkları var).
- Arayüz: Streamlit, `app.py` tek dosya (~470 satır) + `data_engine.py` (ham OHLCV veri çekme/DB kaydetme).
- Depolama: yerel SQLite (`bist_portfolio.db`) — ticker başına ham OHLCV tablosu (`ASELS_IS` vb.) + `watchlist`/`portfoy` tabloları. **Streamlit Cloud'da bu disk geçici (ephemeral)** — redeploy/restart'ta veri kaybolabilir, henüz çözülmedi.

## 2026-07-12: Strateji/AI katmanı tamamen kaldırıldı
Önceki sürümlerde bir teknik analiz motoru (EMA/RSI/MACD/ADX/SuperTrend/StochRSI/Bollinger/OBV indikatörleri, Kısa/Orta/Uzun Vade trend sistemi, Destek/Direnç, Fibonacci, backtest, 3 panelli Plotly grafiği) ve üç AI entegrasyonu (Gemini, Claude, Groq) vardı. Backtest'i 2 yıllık veriyle derinlemesine test ettikten sonra basit "al-tut"un stratejileri büyük farkla geçtiği görüldü (bkz. git geçmişi) — kullanıcı kararıyla **tüm strateji/indikatör/backtest/grafik katmanı ve tüm AI entegrasyonları (Claude/Groq/Anthropic/Tavily) koddan tamamen silindi**. `data_engine.py` artık sadece ham OHLCV çekip DB'ye yazıyor, indikatör hesaplamıyor.

Kalan tek sekme: **💼 Portföy** (adet/maliyet/kâr-zarar takibi). Üstte Watchlist Yönetimi ve Günlük Özet (sadece yoğunlaşma/çeşitlilik uyarısı — pozisyon büyüklüğü bazlı, teknik göstergeden bağımsız) var. Portföydeki stop-loss/hedef disiplin kontrolü de (destek/direnç'e bağımlı olduğu için) kaldırıldı.

**Yeniden tasarım bekleniyor** — kullanıcı yeni bir strateji/karar destek yaklaşımı üzerinde ayrı bir oturumda çalışacak. Bu konuda geçmiş konuşmaları referans almadan önce mevcut `app.py`'yi oku, mimari köklü şekilde değişmiş olabilir.

## Risk Yönetimi Katmanı
Sadece yoğunlaşma uyarısı (tek pozisyon %35'i geçerse) ve çeşitlilik uyarısı (<3 pozisyon) kaldı — `_portfoy_risk_uyarilari()`. Teknik gösterge bazlı stop-loss/hedef kontrolü yok.

## Canlı Fiyat
Günlük mum verisi (`history()`) bazen o günün OHLC'sini NaN bırakıyor veya bir gün gecikiyor — bu yüzden `get_canli_fiyat()` `fast_info` üzerinden anlık fiyatı ayrıca çekiyor (60sn cache), başarısız olursa son kapanışa düşüyor. Fiyatın yanında kontrol saati (HH:MM) gösteriliyor.

## Arayüz Yapısı
Tek sekme: 💼 Portföy. Üstünde her zaman görünen Günlük Özet kutusu (yoğunlaşma uyarısı) ve Watchlist Yönetimi var.

## Doğrulama Alışkanlığı (bu proje için önemli)
- `python -m py_compile app.py data_engine.py` her değişiklikten sonra.
- Saf fonksiyonları gerçek ASELS/AKBNK verisiyle scratchpad'de izole test et (app.py'yi doğrudan import etme — üst seviyede DB bağlantısı/`st.stop()` çalıştırıyor).
- **Uçtan uca doğrulama için `streamlit.testing.v1.AppTest` kullan** — düz bir `streamlit run` + HTTP GET script'i gerçekten ÇALIŞTIRMAZ, sadece statik sayfa döner. `AppTest.from_file(...).run()` + `at.exception` kontrolü tek güvenilir yöntem.
- DB şemasına yeni kolon eklerken `load_data()`'nın self-heal mekanizması sadece "tablo yok" hatasında devreye giriyor, mevcut tabloya otomatik kolon eklemiyor/silmiyor — şema değiştiğinde eski ticker tablolarını `DROP TABLE` ile silip yeniden oluşturmak gerekiyor (watchlist/portfoy tablolarına dokunmadan).
