from modules.database import init_db, ogrenci_ekle_veya_getir, deneme_kaydet
from modules.analysis import konu_bazli_analiz, zayif_konular, genel_ozet
from modules.recommender import kisisel_oneri_uret

# 1. Veritabanını başlat
print("Veritabanı başlatılıyor...")
init_db()

# 2. Test öğrencisi ekle
ogrenci_id = ogrenci_ekle_veya_getir("Ceyda Gülen", "21360859042")
print(f"Öğrenci ID: {ogrenci_id}")

# 3. Sahte quiz ve cevaplar oluştur (test için)
sahte_quiz = [
    {"soru": "Random Forest nedir?", "konu": "Random Forest", "dogru_cevap": "A"},
    {"soru": "Karar ağacı nasıl çalışır?", "konu": "Karar Ağaçları", "dogru_cevap": "B"},
    {"soru": "Gini indeksi nedir?", "konu": "Karar Ağaçları", "dogru_cevap": "C"},
    {"soru": "SVM kernel trick nedir?", "konu": "SVM", "dogru_cevap": "A"},
    {"soru": "Cross-validation nedir?", "konu": "Model Değerlendirme", "dogru_cevap": "D"},
]

# Öğrenci bazı soruları yanlış yapsın (test senaryosu)
sahte_cevaplar = {
    0: "A",  # Random Forest - DOĞRU
    1: "A",  # Karar Ağaçları - YANLIŞ (doğru B)
    2: "A",  # Karar Ağaçları - YANLIŞ (doğru C)
    3: "A",  # SVM - DOĞRU
    4: "D",  # Model Değerlendirme - DOĞRU
}

# 4. Denemeyi kaydet
deneme_id = deneme_kaydet(ogrenci_id, "Makine Öğrenmesi", sahte_quiz, sahte_cevaplar)
print(f"Deneme kaydedildi, ID: {deneme_id}")

# 5. Analiz yap
print("\n--- KONU BAZLI ANALİZ ---")
analiz = konu_bazli_analiz(ogrenci_id)
for konu, veri in analiz.items():
    print(f"{konu}: %{veri['basari_orani']} ({veri['dogru']}/{veri['toplam']})")

print("\n--- ZAYIF KONULAR ---")
print(zayif_konular(ogrenci_id))

print("\n--- GENEL ÖZET ---")
print(genel_ozet(ogrenci_id))

print("\n--- KİŞİSEL ÖNERİ ---")
print(kisisel_oneri_uret(ogrenci_id, "Ceyda"))