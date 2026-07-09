import sqlite3

# Veri tabanına bağlanıyoruz
conn = sqlite3.connect('bist_portfolio.db')
cursor = conn.cursor()

# İçerideki tabloları çekiyoruz
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tablolar = cursor.fetchall()

print("\n--- VERİ TABANI KONTROLÜ ---")
if not tablolar:
    print("❌ İçeride henüz hiç tablo oluşmamış kanka.")
else:
    print("✅ Başarıyla oluşturulan tabloların:")
    for tablo in tablolar:
        print(f"-> {tablo[0]}")
print("---------------------------\n")

conn.close()