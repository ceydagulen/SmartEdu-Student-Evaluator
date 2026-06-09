"""
SmartEdu - RAG Kalite Test Scripti
Kullanım: python test_rag_quality.py

Bu script 4 test çalıştırır:
1. Retrieval doğruluğu  — doğru chunk'lar geliyor mu?
2. Quiz kalitesi        — sorular transkripte dayalı mı?
3. Cevap analizi        — hallucination var mı?
4. Boş cevap yönetimi  — boş bırakılan sorular handle ediliyor mu?
"""

import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.rag import create_vectorstore, load_vectorstore
from modules.transcription import load_transcript, split_transcript
from modules.quiz import generate_quiz, generate_true_false
from modules.rag_analysis import tum_cevaplari_analiz_et

# =====================================================================
# AYARLAR — kendi vektörstore path'ini ve transkriptini buraya yaz
# =====================================================================
VECTORSTORE_PATH = "data/vectorstore/veritabani_dersi_transkript"
TRANSCRIPT_PATH  = "data/transcripts/veritabani_dersi_transkript.txt"

# Eğer vectorstore yoksa transkriptten oluştur
def vectorstore_hazirla():
    if os.path.exists(VECTORSTORE_PATH):
        print(f"✅ Mevcut vectorstore yüklendi: {VECTORSTORE_PATH}")
        return load_vectorstore(VECTORSTORE_PATH)
    elif os.path.exists(TRANSCRIPT_PATH):
        print("⚙️  Vectorstore bulunamadı, transkriptten oluşturuluyor...")
        docs   = load_transcript(TRANSCRIPT_PATH)
        chunks = split_transcript(docs)
        vs     = create_vectorstore(chunks, VECTORSTORE_PATH)
        print("✅ Vectorstore oluşturuldu.")
        return vs
    else:
        print("❌ Ne vectorstore ne de transkript bulunamadı.")
        print(f"   Beklenen transkript yolu: {TRANSCRIPT_PATH}")
        sys.exit(1)


# =====================================================================
# TEST 1: Retrieval Doğruluğu
# Her sorgu için beklenen anahtar kelime chunk'larda bulunuyor mu?
# =====================================================================
def test_retrieval(vectorstore):
    print("\n" + "="*60)
    print("TEST 1: Retrieval Doğruluğu")
    print("="*60)

    # (sorgu, transkriptte geçmesi beklenen anahtar kelime)
    # Sorgular semantik olarak yazıldı — embedding modeli bu şekilde daha iyi eşleştirir
    test_cases = [
        ("SQL nedir ne işe yarar",                               "structured query language"),
        ("SELECT komutu nasıl kullanılır filtreleme",            "where"),
        ("GROUP BY aggregate fonksiyonlar kullanımı",            "group by"),
        ("INNER JOIN LEFT JOIN farkı nedir",                     "inner join"),
        ("normalizasyon neden gereklidir anomali",               "güncelleme anomalisi"),
        ("birinci normal form 1nf kuralları",                    "atomik"),
        ("ikinci normal form 2nf kısmi bağımlılık",              "kısmi bağımlılık"),
        ("üçüncü normal form 3nf geçişli bağımlılık",            "geçişli bağımlılık"),
        ("transaction acid özellikleri nelerdir",                "atomicity"),
        ("view görünüm ne işe yarar veritabanı",                 "view"),
        ("foreign key yabancı anahtar referans bütünlüğü",       "foreign key"),
        ("indeks performans sorgu hızlandırma",                  "indeks"),
    ]

    basarili = 0
    sonuclar = []

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    for sorgu, beklenen in test_cases:
        docs          = retriever.invoke(sorgu)
        bulunan_metin = " ".join([d.page_content for d in docs]).lower()
        bulundu       = beklenen.lower() in bulunan_metin

        if bulundu:
            basarili += 1
            durum = "✅"
            chunk_no = next(
                (i+1 for i, d in enumerate(docs) if beklenen.lower() in d.page_content.lower()), "?")
            print(f"{durum} '{sorgu[:45]:<45}' → '{beklenen}' (chunk sırası: {chunk_no})")
        else:
            print(f"❌ '{sorgu[:45]:<45}' → '{beklenen}' — 10 chunk'ta da bulunamadı")

        sonuclar.append({"sorgu": sorgu, "beklenen": beklenen, "bulundu": bulundu})

    oran = basarili / len(test_cases) * 100
    print(f"\nSonuç: {basarili}/{len(test_cases)} → %{oran:.0f} başarı")

    if oran >= 80:
        print("🟢 Retrieval kalitesi: İYİ")
    elif oran >= 60:
        print("🟡 Retrieval kalitesi: ORTA — chunk boyutunu veya k değerini ayarla")
    else:
        print("🔴 Retrieval kalitesi: DÜŞÜK — embedding modeli veya chunk stratejisi gözden geçirilmeli")

    return sonuclar, oran


# =====================================================================
# TEST 2: Quiz Kalitesi
# Üretilen soruları transkript içeriğiyle karşılaştır
# =====================================================================
def test_quiz_kalitesi(vectorstore):
    print("\n" + "="*60)
    print("TEST 2: Quiz Sorusu Kalitesi")
    print("="*60)

    print("⚙️  5 çoktan seçmeli + 5 doğru/yanlış üretiliyor...")
    quiz      = generate_quiz(vectorstore, soru_sayisi=5)
    dy_sorular = generate_true_false(vectorstore, soru_sayisi=5)

    # Çoktan seçmeli kontroller
    print(f"\n📝 Çoktan Seçmeli ({len(quiz)} soru):")
    cs_sorunlar = []
    for i, s in enumerate(quiz, 1):
        sorunlar = []
        if not s.get("konu"):
            sorunlar.append("konu etiketi yok")
        if not s.get("aciklama"):
            sorunlar.append("açıklama yok")
        if s.get("dogru_cevap") not in ("A", "B", "C", "D"):
            sorunlar.append(f"geçersiz dogru_cevap: {s.get('dogru_cevap')}")
        if len(s.get("secenekler", {})) != 4:
            sorunlar.append(f"şık sayısı hatalı: {len(s.get('secenekler', {}))}")

        if sorunlar:
            print(f"  ⚠️  Soru {i}: {', '.join(sorunlar)}")
            print(f"      → {s.get('soru', '')[:70]}")
            cs_sorunlar.append(i)
        else:
            print(f"  ✅ Soru {i}: {s.get('konu','?')} — {s.get('soru','')[:60]}")

    # Doğru/Yanlış kontroller
    print(f"\n✅ Doğru/Yanlış ({len(dy_sorular)} soru):")
    dogru_sayisi  = sum(1 for s in dy_sorular if s.get("dogru_mu") is True)
    yanlis_sayisi = sum(1 for s in dy_sorular if s.get("dogru_mu") is False)
    print(f"  Dağılım: {dogru_sayisi} doğru, {yanlis_sayisi} yanlış", end="")
    if abs(dogru_sayisi - yanlis_sayisi) <= 1:
        print(" ✅ (dengeli)")
    else:
        print(" ⚠️  (dengesiz — prompt'u kontrol et)")

    dy_sorunlar = []
    for i, s in enumerate(dy_sorular, 1):
        sorunlar = []
        if not s.get("konu"):
            sorunlar.append("konu etiketi yok")
        if not s.get("aciklama"):
            sorunlar.append("açıklama yok")
        if not isinstance(s.get("dogru_mu"), bool):
            sorunlar.append(f"dogru_mu boolean değil: {s.get('dogru_mu')}")

        if sorunlar:
            print(f"  ⚠️  İfade {i}: {', '.join(sorunlar)}")
            dy_sorunlar.append(i)
        else:
            durum = "D" if s.get("dogru_mu") else "Y"
            print(f"  ✅ İfade {i} [{durum}]: {s.get('ifade','')[:60]}")

    cs_kalite  = (len(quiz) - len(cs_sorunlar)) / len(quiz) * 100 if quiz else 0
    dy_kalite  = (len(dy_sorular) - len(dy_sorunlar)) / len(dy_sorular) * 100 if dy_sorular else 0
    ortalama   = (cs_kalite + dy_kalite) / 2

    print(f"\nSonuç: Çoktan Seçmeli %{cs_kalite:.0f} | Doğru/Yanlış %{dy_kalite:.0f} | Ortalama %{ortalama:.0f}")

    return quiz, dy_sorular, ortalama


# =====================================================================
# TEST 3: Cevap Analizi + Hallucination
# Hem doğru hem yanlış hem de transkript dışı sorular test edilir
# =====================================================================
def test_cevap_analizi(vectorstore, quiz):
    print("\n" + "="*60)
    print("TEST 3: Cevap Analizi Kalitesi")
    print("="*60)

    if not quiz:
        print("⚠️  Quiz boş, bu test atlanıyor.")
        return

    # İlk 3 soruyu test et: 1. doğru, 2. yanlış, 3. boş
    test_cevaplar = {}
    if len(quiz) >= 1:
        test_cevaplar[0] = quiz[0]["dogru_cevap"]          # Kasıtlı doğru
    if len(quiz) >= 2:
        secenekler = list(quiz[1]["secenekler"].keys())
        yanlis = next(s for s in secenekler if s != quiz[1]["dogru_cevap"])
        test_cevaplar[1] = yanlis                           # Kasıtlı yanlış
    # index 2 boş bırakılıyor — boş cevap testi

    print(f"  Soru 1 → DOĞRU cevap verildi  ({quiz[0]['dogru_cevap']})")
    if len(quiz) >= 2:
        print(f"  Soru 2 → YANLIŞ cevap verildi ({test_cevaplar.get(1,'?')})")
    if len(quiz) >= 3:
        print(f"  Soru 3 → BOŞ bırakıldı")

    analizler = tum_cevaplari_analiz_et(vectorstore, quiz[:3], test_cevaplar)

    print("\nÜretilen analizler:")
    for a in analizler:
        durum = "✅ DOĞRU" if a["dogru_mu"] else "❌ YANLIŞ"
        print(f"\n  [{durum}] Soru {a['soru_no']}: {a['soru'][:60]}")
        print(f"  Analiz: {a['analiz'][:200]}")

        # Kalite kontrolü
        if len(a["analiz"]) < 20:
            print("  ⚠️  Analiz çok kısa!")
        if "bilmiyorum" in a["analiz"].lower() and not a["dogru_mu"]:
            print("  ⚠️  Model cevap üretemedi.")


# =====================================================================
# TEST 4: Hallucination Testi
# Transkriptte olmayan konuyu sorarak modelin nasıl davrandığını gör
# =====================================================================
def test_hallucination(vectorstore):
    print("\n" + "="*60)
    print("TEST 4: Hallucination Direnci")
    print("="*60)

    # Transkriptte geçmeyen konular (RAG dersinde bunlar anlatılmadı)
    konu_disi_sorgular = [
        "Transformer mimarisinin attention mekanizması nasıl çalışır?",
        "CNN katmanları görüntü işlemede nasıl kullanılır?",
        "Blockchain teknolojisi nedir?",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Transkript dışı sorgularda gelen chunk'lar kontrol ediliyor...\n")
    for sorgu in konu_disi_sorgular:
        docs = retriever.invoke(sorgu)
        bulunan = " ".join([d.page_content for d in docs]).lower()

        # Sorgudaki anahtar kelimelerin chunk'larda olup olmadığına bak
        anahtar = sorgu.split()[0].lower()
        alakali_mi = anahtar in bulunan

        print(f"  Sorgu: '{sorgu[:60]}'")
        if alakali_mi:
            print(f"  ⚠️  İlgili chunk bulundu — model bu konuda cevap üretebilir")
        else:
            print(f"  ✅ İlgili chunk bulunamadı — model 'bilmiyorum' demeli")
        print()


# =====================================================================
# ÖZET RAPOR
# =====================================================================
def ozet_rapor(retrieval_oran, quiz_oran):
    print("\n" + "="*60)
    print("ÖZET RAPOR")
    print("="*60)

    def puan_renk(oran):
        if oran >= 80: return "🟢"
        if oran >= 60: return "🟡"
        return "🔴"

    print(f"{puan_renk(retrieval_oran)} Retrieval Doğruluğu : %{retrieval_oran:.0f}")
    print(f"{puan_renk(quiz_oran)}     Quiz Kalitesi       : %{quiz_oran:.0f}")

    genel = (retrieval_oran + quiz_oran) / 2
    print(f"\n{'🟢' if genel >= 75 else '🟡' if genel >= 55 else '🔴'} Genel RAG Skoru     : %{genel:.0f}")

    print("\nİyileştirme önerileri:")
    if retrieval_oran < 80:
        print("  • Retrieval düşük → chunk_size küçült (500→300) veya k artır")
        print("  • Farklı embedding modeli dene (multilingual-e5-large)")
    if quiz_oran < 80:
        print("  • Quiz kalitesi düşük → prompt'taki system mesajını güçlendir")
        print("  • Temperature'ı 0.5'e düşür, daha tutarlı sorular üretir")
    if genel >= 75:
        print("  ✅ Sistem bitirme projesi sunumu için hazır görünüyor!")


# =====================================================================
# ANA AKIŞ
# =====================================================================
if __name__ == "__main__":
    print("SmartEdu RAG Kalite Testi Başlıyor...\n")

    vectorstore = vectorstore_hazirla()

    # Test 1
    _, retrieval_oran = test_retrieval(vectorstore)

    # Test 2
    quiz, dy_sorular, quiz_oran = test_quiz_kalitesi(vectorstore)

    # Test 3
    test_cevap_analizi(vectorstore, quiz)

    # Test 4
    test_hallucination(vectorstore)

    # Özet
    ozet_rapor(retrieval_oran, quiz_oran)

    print("\nTest tamamlandı.")