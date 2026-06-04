from modules.rag import load_vectorstore
from modules.rag_analysis import tum_cevaplari_analiz_et

# Vektör veritabanını yükle
print("Vektör veritabanı yükleniyor...")
vectorstore = load_vectorstore("data/vectorstore/nyp_1_02")

# 5 soruluk test quizi
test_quiz = [
    {
        "soru": "String veri tipi nedir?",
        "konu": "String",
        "secenekler": {
            "A": "Bir sayısal veri tipidir",
            "B": "Bir karakter dizisidir",
            "C": "Bir boolean değerdir",
            "D": "Bir döngü yapısıdır"
        },
        "dogru_cevap": "B"
    },
    {
        "soru": "Primitif veri tipleri nasıl başlar?",
        "konu": "Değişkenler",
        "secenekler": {
            "A": "Büyük harfle",
            "B": "Küçük harfle",
            "C": "Rakamla",
            "D": "Alt çizgiyle"
        },
        "dogru_cevap": "B"
    },
    {
        "soru": "Tırnak işaretleri ne için kullanılır?",
        "konu": "String",
        "secenekler": {
            "A": "Sayılar için",
            "B": "Döngüler için",
            "C": "Karakter dizileri için",
            "D": "Fonksiyonlar için"
        },
        "dogru_cevap": "C"
    },
    {
        "soru": "Bir String'i integer'a çeviren fonksiyon hangisidir?",
        "konu": "Tip Dönüşümü",
        "secenekler": {
            "A": "atoi",
            "B": "print",
            "C": "input",
            "D": "while"
        },
        "dogru_cevap": "A"
    },
    {
        "soru": "Değişken nedir?",
        "konu": "Değişkenler",
        "secenekler": {
            "A": "Bir döngü türü",
            "B": "Veri saklamak için kullanılan yapı",
            "C": "Bir hata mesajı",
            "D": "Bir operatör"
        },
        "dogru_cevap": "B"
    }
]

# Öğrenci cevapları (bazıları doğru, bazıları yanlış)
ogrenci_cevaplari = {
    0: "A",  # YANLIŞ (doğru B)
    1: "B",  # DOĞRU
    2: "C",  # DOĞRU
    3: "C",  # YANLIŞ (doğru A)
    4: "B",  # DOĞRU
}

# Tüm cevapları analiz et
print("\nCevaplar RAG ile analiz ediliyor...\n")
sonuclar = tum_cevaplari_analiz_et(vectorstore, test_quiz, ogrenci_cevaplari)

for s in sonuclar:
    durum = "✅ DOĞRU" if s["dogru_mu"] else "❌ YANLIŞ"
    print(f"Soru {s['soru_no']}: {s['soru']}")
    print(f"  Durum: {durum} (Konu: {s['konu']})")
    print(f"  Analiz: {s['analiz']}")
    print("-" * 60)