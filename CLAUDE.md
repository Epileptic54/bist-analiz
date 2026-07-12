# BIST Dashboard Projesi Kuralları

## Proje Amacı
Kişisel bir BIST (Borsa İstanbul) portföy takip panelim. Watchlist dinamik (SQLite'ta saklanıyor, arayüzden ekle/çıkar) — sabit bir hisse listesi yok, varsayılan olarak ASELS/ASTOR/THYAO/BIMAS/AKBNK ile başladı ama kullanıcı istediği hisseyi ekleyebiliyor. Portföy takibi (adet/maliyet/kâr-zarar) ana özellik.

## Teknik Yığın
- Backend/veri: Python, `yfinance` (ücretsiz, resmi olmayan kaynak — bilinen kırılganlıkları var).
- Arayüz: Streamlit, `app.py` tek dosya (~470 satır) + `data_engine.py` (ham OHLCV veri çekme/DB kaydetme).
- Depolama: yerel SQLite (`bist_portfolio.db`) — ticker başına ham OHLCV tablosu (`ASELS_IS` vb.) + `watchlist`/`portfoy` tabloları. **Streamlit Cloud'da bu disk geçici (ephemeral)** — redeploy/restart'ta veri kaybolabilir, henüz çözülmedi.

## 2026-07-12: Bir günde silindi, yeniden kuruldu, sonra 10 Temmuz haline geri alındı
Aynı gün içinde üç aşama yaşandı: (1) 19:28'de tüm strateji/indikatör/backtest/AI katmanı silindi (`2c6640e`), (2) 20:22-21:11 arası teknik analiz katmanı sıfırdan yeniden kuruldu (`5c5da26`, `fb759aa`, AI hariç), (3) kullanıcı bu deneyden memnun kalmayıp **günün tüm değişikliklerini geri alıp `app.py`/`data_engine.py`'yi 10 Temmuz'daki (`b4d34a8`) haline döndürdü**. Yani şu an kod tabanı, 12 Temmuz'da hiç dokunulmamış gibi — 10 Temmuz'daki eski sistemle aynı.

## Mevcut Sistem (10 Temmuz hali — b4d34a8)
Eski teknik analiz motoru geri geldi: EMA/RSI/MACD/ADX/SuperTrend/StochRSI/Bollinger/OBV indikatörleri, Kısa/Orta/Uzun Vade trend sistemi, Destek/Direnç, Fibonacci, backtest, 3 panelli Plotly grafiği. Üç AI entegrasyonu (Gemini, Claude, Groq) da geri geldi. Portföydeki stop-loss/hedef disiplin kontrolü (destek/direnç'e bağımlı) da geri geldi.

**Not:** 12 Temmuz'daki backtest denemesinde 2 yıllık veriyle basit "al-tut" stratejisinin Vade/Destek-Direnç stratejilerini büyük farkla geçtiği görülmüştü (bkz. git geçmişi, `2c6640e` öncesi commit mesajları). Kullanıcı bu bulguya rağmen eski sisteme dönmeyi tercih etti — ileride tekrar "stratejiyi basitleştirelim" denirse bu geçmişi hatırlat, ama karar kullanıcıya ait.

Yeni değişiklik yapmadan önce mevcut `app.py`'yi oku — bu dosyadaki notlar hızlı değişebiliyor, git log tarihine güven, geçmiş konuşmalara değil.

## Canlı Fiyat
Günlük mum verisi (`history()`) bazen o günün OHLC'sini NaN bırakıyor veya bir gün gecikiyor — bu yüzden `get_canli_fiyat()` `fast_info` üzerinden anlık fiyatı ayrıca çekiyor (60sn cache), başarısız olursa son kapanışa düşüyor. Fiyatın yanında kontrol saati (HH:MM) gösteriliyor.

## Doğrulama Alışkanlığı (bu proje için önemli)
- `python -m py_compile app.py data_engine.py` her değişiklikten sonra.
- Saf fonksiyonları gerçek ASELS/AKBNK verisiyle scratchpad'de izole test et (app.py'yi doğrudan import etme — üst seviyede DB bağlantısı/`st.stop()` çalıştırıyor).
- **Uçtan uca doğrulama için `streamlit.testing.v1.AppTest` kullan** — düz bir `streamlit run` + HTTP GET script'i gerçekten ÇALIŞTIRMAZ, sadece statik sayfa döner. `AppTest.from_file(...).run()` + `at.exception` kontrolü tek güvenilir yöntem.
- DB şemasına yeni kolon eklerken `load_data()`'nın self-heal mekanizması sadece "tablo yok" hatasında devreye giriyor, mevcut tabloya otomatik kolon eklemiyor/silmiyor — şema değiştiğinde eski ticker tablolarını `DROP TABLE` ile silip yeniden oluşturmak gerekiyor (watchlist/portfoy tablolarına dokunmadan).
