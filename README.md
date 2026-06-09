# 🎓 SmartEdu — Yapay Zeka Destekli Öğrenci Değerlendirme Sistemi

Ders transkriptlerini analiz ederek otomatik quiz oluşturan, öğrenci cevaplarını değerlendiren ve kişiselleştirilmiş öğrenme önerileri sunan yapay zeka destekli eğitim asistanı.

> Bitirme Projesi · Bursa Teknik Üniversitesi · Bilgisayar Mühendisliği

---

## ✨ Özellikler

- **Esnek içerik girişi** — YouTube video linki, ses dosyası veya Zoom/Meet/Teams transkriptini doğrudan yapıştırma
- **Otomatik transkripsiyon** — Whisper ile ses → metin dönüşümü
- **Kavram haritası** — Dersin ana konuları, alt başlıkları ve anahtar kavramları otomatik çıkarılır
- **Dinamik quiz üretimi** — Çoktan seçmeli ve doğru/yanlış soruları, konu etiketleriyle birlikte
- **RAG tabanlı cevap analizi** — Öğrenci cevapları transkripte göre bağlamsal olarak değerlendirilir
- **Performans takibi** — Konu bazlı başarı, gelişim grafiği ve kişiselleştirilmiş öğrenme önerileri

---

## 🛠️ Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Arayüz | Streamlit |
| LLM | Groq API (LLaMA 3.3 70B) |
| Embedding | OpenAI (text-embedding-3-small) |
| RAG | LangChain + ChromaDB |
| Transkripsiyon | OpenAI Whisper |
| Veritabanı | SQLite |

---

## 🚀 Kurulum

**1. Projeyi klonla**
```bash
git clone https://github.com/ceydagulen/SmartEdu-Student-Evaluator.git
cd SmartEdu-Student-Evaluator
```

**2. Sanal ortam oluştur ve etkinleştir**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

**3. Gerekli paketleri yükle**
```bash
pip install -r requirements.txt
```

**4. `.env` dosyası oluştur**
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

**5. Uygulamayı başlat**
```bash
streamlit run app.py
```

> **Not:** YouTube video özelliği için `ffmpeg` ve `yt-dlp` sistemde kurulu olmalıdır.

---

## 📖 Kullanım

1. Öğrenci girişi yap (ad-soyad ve numara)
2. Ders kaynağı seç: video linki, transkript yapıştır veya kayıtlı ders
3. Quiz tipini ve soru sayısını belirle
4. Soruları cevapla ve gönder
5. Sonuç ekranında puanını, cevap analizini ve öğrenme önerilerini gör

---

## 📁 Proje Yapısı

```
SmartEdu-Student-Evaluator/
├── app.py                    # Streamlit arayüzü ve ana akış
├── modules/
│   ├── transcription.py      # Whisper transkripsiyon + metin temizleme
│   ├── rag.py                # ChromaDB vektör veritabanı + embedding
│   ├── concept_map.py        # Kavram haritası çıkarma
│   ├── quiz.py               # Quiz ve doğru/yanlış üretimi
│   ├── rag_analysis.py       # RAG tabanlı cevap analizi
│   ├── analysis.py           # Performans ve konu bazlı analiz
│   ├── recommender.py        # Kişiselleştirilmiş öneriler
│   └── database.py           # SQLite işlemleri
├── data/                     # Transkriptler ve vektör veritabanı (otomatik oluşur)
├── requirements.txt
└── .env                      # API anahtarları (git'e dahil değil)
```

---

## 👤 Geliştirici

**Ceyda Gülen**
Danışman: Prof. Dr. Turgay Tugay Bilgin
2025-2026 Bahar Dönemi