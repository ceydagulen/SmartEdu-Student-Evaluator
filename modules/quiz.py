from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

load_dotenv()

def generate_quiz(vectorstore, ders_id: str = "ders_1", soru_sayisi: int = 5) -> list:
    """
    Transkriptten otomatik quiz soruları üretir.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    # Önce dersin ana konularını çıkar
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke("bu derste hangi konular anlatıldı ana başlıklar neler")
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Aşağıdaki ders transkriptine dayanarak {soru_sayisi} adet çoktan seçmeli soru üret.

Transkript:
{context}

Kurallar:
- Her soru 4 şık içermeli (A, B, C, D) Her soru için mutlaka bir "konu" etiketi belirle, bu konu dersin alt başlıklarından biri olmalı
- Sorular sadece ezbere değil, anlama ve uygulama odaklı olmalı,Her soru için mutlaka bir "konu" etiketi belirle, bu konu dersin alt başlıklarından biri olmalı
- Bazı sorular "neden", "nasıl", "hangisi yanlış" gibi düşündürücü sorular olmalı,Her soru için mutlaka bir "konu" etiketi belirle, bu konu dersin alt başlıklarından biri olmalı
- Yanlış şıklar mantıklı ve yanıltıcı olmalı, çok bariz olmamalı,Her soru için mutlaka bir "konu" etiketi belirle, bu konu dersin alt başlıklarından biri olmalı
- Her sorunun bir doğru cevabı olmalı,er soru için mutlaka bir "konu" etiketi belirle, bu konu dersin alt başlıklarından biri olmalı
- Türkçe yaz
- Sadece JSON formatında döndür, başka hiçbir şey yazma,Her soru için mutlaka bir "konu" etiketi belirle, bu konu dersin alt başlıklarından biri olmalı
- Eğer transkriptte anlamsız, bozuk veya yanlış yazılmış görünen teknik terimler varsa (örneğin Türkçe ses çevirisinden kaynaklanan hatalar), bunları görmezden gel
- Sadece net ve anlaşılır konulardan soru üret
- Bir terimin doğru olup olmadığından emin değilsen o terimi soruda kullanma
- Soruların eğitici ve mantıklı olmasına öncelik ver
JSON formatı:
[
  {{
    "soru": "Soru metni",
    "konu": "Bu sorunun ait olduğu konu (örn: Random Forest, Karar Ağaçları)",
    "secenekler": {{
      "A": "Seçenek A",
      "B": "Seçenek B", 
      "C": "Seçenek C",
      "D": "Seçenek D"
    }},
    "dogru_cevap": "A",
    "aciklama": "Neden bu cevap doğru, diğerleri neden yanlış"
  }}
]"""

    response = llm.invoke(prompt)
    
    # JSON parse et
    try:
        # Bazen model ```json ``` ile sarıyor, temizle
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        quiz = json.loads(text)
        print(f"{len(quiz)} soru üretildi.")
        return quiz
    except json.JSONDecodeError:
        print("JSON parse hatası, ham cevap:")
        print(response.content)
        return []


def print_quiz(quiz: list):
    """
    Quiz sorularını ekrana yazdırır.
    """
    for i, soru in enumerate(quiz, 1):
        print(f"\nSoru {i}: {soru['soru']}")
        for harf, metin in soru['secenekler'].items():
            print(f"  {harf}) {metin}")
        print(f"  ✅ Doğru cevap: {soru['dogru_cevap']}")
        print(f"  📝 Açıklama: {soru['aciklama']}")

def generate_true_false(vectorstore, soru_sayisi: int = 5) -> list:
    """
    Transkriptten doğru/yanlış soruları üretir.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke("bu derste anlatılan önemli bilgiler ve kavramlar")
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Aşağıdaki ders transkriptine dayanarak {soru_sayisi} adet doğru/yanlış sorusu üret.

Transkript:
{context}

Kurallar:
- Her ifade ya doğru ya yanlış olmalı
- İfadelerin yaklaşık yarısı doğru, yarısı yanlış olsun
- Yanlış ifadeler mantıklı ama hatalı bilgi içersin (öğrenciyi düşündürsün)
- Anlamsız veya bozuk görünen teknik terimleri kullanma
- Türkçe yaz
- Her soruya bir konu etiketi ekle
- Sadece JSON formatında döndür, başka hiçbir şey yazma

JSON formatı:
[
  {{
    "ifade": "Değerlendirilecek ifade",
    "konu": "Konu adı",
    "dogru_mu": true,
    "aciklama": "Neden doğru veya yanlış olduğunun açıklaması"
  }}
]"""

    response = llm.invoke(prompt)

    try:
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        sorular = json.loads(text)
        print(f"{len(sorular)} doğru/yanlış sorusu üretildi.")
        return sorular
    except json.JSONDecodeError:
        print("JSON parse hatası:")
        print(response.content)
        return []