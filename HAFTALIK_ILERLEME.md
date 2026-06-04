# Bitirme Projesi Haftalık İlerleme Raporu

## Proje Bilgileri

| Alan | Bilgi |
|------|-------|
| **Öğrenci Adı Soyadı** | *CEYDA GÜLEN* |
| **Öğrenci No** | *21360859042* |
| **Proje Başlığı** | *Yapay Zeka Destekli Öğrenci Değerlendirme ve Kişiselleştirilmiş Öğrenme Öneri Sistemi* |
| **Danışman** | Prof. Dr. Turgay Tugay Bilgin |
| **Dönem** | 2025-2026 Bahar |

---

## İş Planı

> **Kullanım:** Dönem başında aşağıdaki tabloyu projenize göre doldurun. Her hafta için planlanan işi ve o haftanın sonunda projenin tahmini tamamlanma oranını yazın. Dönem ilerledikçe "Durum" sütununu güncelleyin.

| Hafta | Tarih Aralığı | Planlanan İş | Tahmini Tamamlanma (%) | Durum |
|-------|---------------|--------------|------------------------|-------|
| 1 | 06.04 - 12.04 | *Literatür Taraması ve RAG Mimarisi Teknik Araştırması* | %10 | ✅ Tamamlandı |
| 2 | 13.04 - 19.04 | *Veri Seti Hazırlığı ve Ön İşleme Betikleri* | %20 | ✅ Tamamlandı |
| 3 | 27.04 - 03.05 | *Vektör Veritabanı  Kurulumu ve Embedding Testleri* | %30 |✅ Tamamlandı |
| 4 | 04.05 - 10.05 | *RAG Akışının Kurulması* | %40 | ✅ Tamamlandı |
| 5 | 11.05 - 17.05 | *Metinden Kavram Haritası Çıkarma Modülünün Geliştirilmesi* | %50 | ✅ Tamamlandı |
| 6 | 18.05 - 24.05 | *Dinamik Quiz Üretim Algoritmasının Kodlanması* | %60 | ✅ Tamamlandı |
| 7 | 25.05 - 31.05 | *Öğrenci Performans Analizi ve Öneri Mantığının Oluşturulması* | %70 | ✅ Tamamlandı |
| 8 | 01.06 - 07.06 | *Web Arayüzü Geliştirme ve API Entegrasyonu* | %80 | 🔄 Devam Ediyor |
| 9 | 08.06 - 14.06 | *Sistem Testleri, Hata Ayıklama ve Model Optimizasyonu* | %90 | ⬜ Başlamadı |
| 10| 15.06 - 21.06 | *Proje Raporunun Tamamlanması ve Sunum Hazırlığı* | %100 | ⬜ Başlamadı |

**Durum simgeleri:** ⬜ Başlamadı | 🔄 Devam Ediyor | ✅ Tamamlandı | ⚠️ Gecikti

---

## Haftalık İlerleme Kayıtları


---

### Hafta 1 *(Tarih: 06.04.2025 - 12.04.2025)*

**Plandaki hedef:**
- Literatür Taraması ve RAG Mimarisi Teknik Araştırması

**Bu hafta yaptıklarım:**
- RAG (Retrieval-Augmented Generation) mimarisi araştırıldı ve proje yapısına nasıl entegre edileceği belirlendi
- LangChain, LangGraph, ChromaDB, Whisper teknolojileri araştırıldı ve karşılaştırıldı
- Groq API entegrasyonu yapıldı, LLaMA 3.3 70B modeli ile ilk bağlantı testi başarıyla tamamlandı
- Proje klasör yapısı oluşturuldu ve GitHub'a push edildi
- Transkript işleme modülü (transcription.py) geliştirildi ve test edildi
- Sanal ortam (venv) kuruldu, gerekli kütüphaneler yüklendi

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. Literatür taraması tamamlandı, RAG mimarisi teknik olarak araştırıldı ve ilk implementasyon adımları atıldı.

**Karşılaştığım sorunlar / zorluklar:**
- Python 3.14 sürümü kütüphane uyumsuzlukları nedeniyle Python 3.11.9 sürümüne geçildi
- llama3-8b-8192 modeli kullanımdan kalktığı için llama-3.3-70b-versatile modeline geçildi
- PowerShell script çalıştırma kısıtlaması giderildi

**Gelecek hafta hedefim:**
- Veri seti hazırlığı ve ön işleme betiklerinin yazılması
- Embedding modülünün geliştirilmesi
- ChromaDB vektör veritabanı kurulumu ve ilk testlerin yapılması



### Hafta 2 *(Tarih: 13.04.2025 - 19.04.2025)*

**Plandaki hedef:**
- Veri Seti Hazırlığı ve Ön İşleme Betikleri

**Bu hafta yaptıklarım:**
- Örnek ders transkripti (ders1.txt) oluşturuldu
- Transkript işleme modülü tamamlandı (transcription.py)
- HuggingFace embedding modeli entegre edildi
- ChromaDB vektör veritabanı kuruldu ve test edildi
- RAG modülü geliştirildi (rag.py)
- "Lineer regresyon nedir?" sorusuna transkriptten başarılı cevap alındı

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. Veri seti hazırlandı, ön işleme betikleri yazıldı ve RAG sistemi başarıyla test edildi.

**Karşılaştığım sorunlar / zorluklar:**
- LangChain yeni versiyonunda bazı modüller taşınmıştı, import yolları güncellendi
- sentence-transformers ve langchain-chroma ek kurulum gerektirdi

**Gelecek hafta hedefim:**
- Vektör veritabanı kurulumunu kalıcı hale getirme
- Embedding testlerini genişletme
- Daha fazla transkript verisiyle sistemin test edilmesi

### Hafta 3 *(Tarih: 27.04.2025 - 03.05.2025)*

**Plandaki hedef:**
- Vektör Veritabanı Kurulumu ve Embedding Testleri

**Bu hafta yaptıklarım:**
- İkinci ders transkripti oluşturuldu (ders2.txt - sınıflandırma algoritmaları)
- Birden fazla transkriptle embedding testi yapıldı
- Çoklu transkript yükleme ve birleştirme sistemi geliştirildi (test_multi_rag.py)
- Farklı derslerden sorulara başarılı cevaplar alındı
- requirements.txt güncellendi

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. Vektör veritabanı kurulumu tamamlandı, embedding testleri başarıyla gerçekleştirildi.

**Karşılaştığım sorunlar / zorluklar:**
- .env dosyasının silinmesi nedeniyle Groq API anahtarı yenilendi

**Gelecek hafta hedefim:**
- RAG akışını daha sağlam hale getirme
- Kavram haritası çıkarma modülünün geliştirilmesi




### Hafta 4 *(Tarih: 04.05.2025 - 10.05.2025)*

**Plandaki hedef:**
- RAG Akışının Kurulması

**Bu hafta yaptıklarım:**
- YouTube'dan ses indirme ve Whisper ile transkripte çevirme modülü geliştirildi
- ffmpeg sisteme kuruldu ve PATH'e eklendi
- Gerçek ders videosu (BTÜ Java dersi) Whisper ile transkripte çevrildi
- Transkript temizleme fonksiyonu geliştirildi (clean_transcript)
- Ders bazlı vektör veritabanı yapısı kuruldu (data/vectorstore/ders_adi/)
- Quiz üretim modülü geliştirildi (modules/quiz.py)
- Streamlit web arayüzü geliştirildi (app.py)
- Web arayüzünde URL girişi ile YouTube/Drive/Zoom/Teams desteği eklendi
- Uçtan uca test yapıldı: URL → transkript → RAG → quiz akışı çalıştı

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. RAG akışı kuruldu ve web arayüzüyle entegre edildi. Quiz üretimi de bu hafta tamamlandığından Hafta 6 hedefi de kısmen karşılandı.

**Karşılaştığım sorunlar / zorluklar:**
- ffmpeg PATH tanımlamasında sorun yaşandı, manuel olarak eklendi
- Whisper CPU'da yavaş çalıştığından uzun videolarda işlem süresi uzadı

**Gelecek hafta hedefim:**
- Kavram haritası çıkarma modülünün geliştirilmesi
- Öğrenci performans analizi modülünün yazılması
- Web arayüzünün geliştirilmesi


### Hafta 5 *(Tarih: 11.05.2025 - 17.05.2025)*

**Plandaki hedef:**
- Metinden Kavram Haritası Çıkarma Modülünün Geliştirilmesi

**Bu hafta yaptıklarım:**
- Kavram haritası çıkarma modülü geliştirildi (concept_map.py)
- Transkriptten ana konular ve alt konular otomatik çıkarılıyor
- Kavram haritası web arayüzüne entegre edildi (metin + görsel ağaç yapısı)
- Groq LLM ile transkript içeriği yapılandırılmış JSON formatına dönüştürüldü

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. Kavram haritası modülü tamamlandı ve test edildi.

**Karşılaştığım sorunlar / zorluklar:**
- Whisper transkript hataları kavram haritası kalitesini etkiledi, çözüm için medium modele geçiş planlandı

**Gelecek hafta hedefim:**
- Dinamik quiz üretim algoritmasının geliştirilmesi


### Hafta 6 *(Tarih: 18.05.2025 - 24.05.2025)*

**Plandaki hedef:**
- Dinamik Quiz Üretim Algoritmasının Kodlanması

**Bu hafta yaptıklarım:**
- Quiz üretim modülü geliştirildi ve iyileştirildi (quiz.py)
- Her soruya konu etiketi eklendi (konu bazlı analiz için temel oluşturuldu)
- Quiz prompt'una kalite filtresi eklendi (hatalı transkript terimlerini görmezden gelme)
- İnteraktif quiz arayüzü tamamlandı (cevap seçimi, doğru/yanlış gösterimi, puanlama)

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. Dinamik quiz üretimi çalışıyor ve web arayüzüne entegre edildi.

**Karşılaştığım sorunlar / zorluklar:**
- Whisper hatalı transkriptlerden anlamsız sorular üretiliyordu, prompt filtresi ve medium model ile çözüldü

**Gelecek hafta hedefim:**
- Öğrenci performans analizi ve öneri sisteminin geliştirilmesi


### Hafta 7 *(Tarih: 25.05.2025 - 31.05.2025)*

**Plandaki hedef:**
- Öğrenci Performans Analizi ve Öneri Mantığının Oluşturulması

**Bu hafta yaptıklarım:**
- SQLite veritabanı kuruldu (database.py): öğrenciler, denemeler ve cevaplar tabloları oluşturuldu
- Konu bazlı performans analizi modülü geliştirildi (analysis.py): zayıf konu tespiti, gelişim takibi, genel özet
- RAG ile cevap analizi modülü geliştirildi (rag_analysis.py): öğrenci cevapları transkripte göre bağlamsal olarak analiz ediliyor
- Kişiselleştirilmiş öneri ve ders önerisi modülü geliştirildi (recommender.py)
- Tüm modüller web arayüzünde birleştirildi (öğrenci girişi → quiz → sonuç → RAG analizi → kişisel öneri)
- Öğrenci gelişim grafiği ve performans paneli eklendi

**Plana göre durumum:**
- Plandaki hedefe ulaşıldı. Proje özetinde belirtilen tüm temel çıktılar (kavram haritası, otomatik quiz, RAG ile cevap analizi, kişiselleştirilmiş öneri) çalışır durumda.

**Karşılaştığım sorunlar / zorluklar:**
- Quiz sonuçlarının kalıcı saklanması için SQLite veritabanı entegrasyonu gerekti
- Konu bazlı analiz için quiz sorularına konu etiketi eklenmesi gerekti

**Gelecek hafta hedefim:**
- Web arayüzünün iyileştirilmesi ve kullanıcı deneyiminin geliştirilmesi
- Sistem testleri ve hata ayıklama


---

<!--
ŞABLON: Yeni hafta eklemek için aşağıdaki bloğu kopyalayıp üste yapıştırın.

### Hafta X *(Tarih: GG.AA.YYYY - GG.AA.YYYY)*

**Plandaki hedef:**
- 

**Bu hafta yaptıklarım:**
- 

**Plana göre durumum:**
- 

**Karşılaştığım sorunlar / zorluklar:**
- 

**Gelecek hafta hedefim:**
- 

---
-->