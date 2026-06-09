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
        model="llama-3.3-70b-versatile",
        temperature=0.7,          # Çeşitlilik için — 0 olursa tekrarcı, 1+ olursa saçmalar
        max_tokens=4096,           # Çok soru üretince cevap kesilebiliyordu
    )

    # Farklı konulardan chunk çekmek için birden fazla query kullan
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    queries = [
        "bu derste anlatılan ana kavramlar ve tanımlar",
        "bu derste verilen örnekler ve uygulamalar",
        "bu derste açıklanan yöntemler ve algoritmalar",
    ]
    
    seen = set()
    all_docs = []
    for q in queries:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    
    context = "\n\n".join([doc.page_content for doc in all_docs])

    system_prompt = """Sen deneyimli bir eğitim uzmanısın. Ders transkriptlerinden yüksek kaliteli, pedagojik değeri olan quiz soruları üretiyorsun.
Görevin: Öğrencilerin dersi gerçekten anlayıp anlamadığını ölçen sorular hazırlamak.
Kural: SADECE verilen JSON formatında cevap ver. Hiçbir ek açıklama, giriş cümlesi veya yorum yazma."""

    user_prompt = f"""Aşağıdaki ders transkriptine dayanarak TAM OLARAK {soru_sayisi} adet çoktan seçmeli soru üret.

TRANSKRIPT:
{context}

SORU KALİTESİ KURALLARI:
1. Sorular ezbere değil, kavrama ve uygulama odaklı olmalı
2. En az 2 soru "neden", "nasıl" veya "hangisi yanlıştır" formatında olmalı
3. Yanlış şıklar mantıklı ve yanıltıcı olmalı — çok bariz olmamalı
4. Her sorunun TEK bir doğru cevabı olmalı
5. Transkriptte bozuk/hatalı görünen terimler varsa o konudan soru üretme
6. Her soru farklı bir konudan olmalı, tekrar etme

ZORUNLU ALANLAR — Her soruda şunlar mutlaka olmalı:
- "soru": Soru metni (Türkçe)
- "konu": Sorunun ait olduğu ders alt başlığı (örn: "Karar Ağaçları", "Overfitting")
- "secenekler": A, B, C, D şıkları
- "dogru_cevap": Tek harf (A/B/C/D)
- "aciklama": Doğru cevabın neden doğru, diğerlerinin neden yanlış olduğunun kısa açıklaması

ÇIKTI FORMATI — Sadece bu JSON array'i döndür, başka hiçbir şey yazma:
[
  {{
    "soru": "...",
    "konu": "...",
    "secenekler": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "dogru_cevap": "A",
    "aciklama": "..."
  }}
]"""

    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    try:
        text = response.content.strip()
        # Model bazen ```json veya ``` ile sarıyor, temizle
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        quiz = json.loads(text)
        
        # Kalite kontrolü: eksik alan olan soruları filtrele
        required_keys = {"soru", "konu", "secenekler", "dogru_cevap", "aciklama"}
        quiz = [q for q in quiz if required_keys.issubset(q.keys())]
        
        # dogru_cevap gerçekten A/B/C/D mi kontrol et
        quiz = [q for q in quiz if q["dogru_cevap"] in ("A", "B", "C", "D")]
        
        print(f"{len(quiz)} geçerli soru üretildi.")
        return quiz
        
    except json.JSONDecodeError as e:
        print(f"JSON parse hatası: {e}")
        print("Ham cevap:", response.content[:500])
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
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=4096,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    queries = [
        "bu derste anlatılan tanımlar ve kavramlar",
        "bu derste açıklanan yöntemler ve algoritmalar",
        "bu derste verilen örnekler ve sonuçlar",
    ]
    seen = set()
    all_docs = []
    for q in queries:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    context = "\n\n".join([doc.page_content for doc in all_docs])

    # Dağılımı kendimiz hesaplayıp modele söylüyoruz
    yanlis_sayi = soru_sayisi // 2
    dogru_sayi = soru_sayisi - yanlis_sayi

    system_prompt = """Sen deneyimli bir eğitim uzmanısın. Ders transkriptlerinden öğrencinin kavrayışını ölçen doğru/yanlış ifadeleri üretiyorsun.
Görevin: Öğrenciyi gerçekten düşündüren, ezber değil anlama odaklı ifadeler hazırlamak.
Kural: SADECE verilen JSON formatında cevap ver. Hiçbir ek açıklama veya yorum yazma."""

    user_prompt = f"""Aşağıdaki ders transkriptine dayanarak TAM OLARAK {soru_sayisi} adet doğru/yanlış ifadesi üret.

TRANSKRIPT:
{context}

DAĞILIM — Bu dağılıma KESINLIKLE uy:
- TAM OLARAK {dogru_sayi} ifade "dogru_mu": true olmalı
- TAM OLARAK {yanlis_sayi} ifade "dogru_mu": false olmalı
- Doğru ve yanlış ifadeleri birbirine karıştır, hepsini başa ya da sona koyma

İFADE KALİTESİ KURALLARI:
1. Her ifade farklı bir konudan olmalı, tekrar etme
2. Yanlış ifadeler mantıklı ama hatalı bilgi içermeli — çok bariz olmamalı
   Örnek iyi yanlış ifade: "Random Forest'ta ağaçlar sıralı olarak eğitilir" (aslında paralel)
   Örnek kötü yanlış ifade: "Karar ağaçları hiçbir zaman kullanılmaz" (çok bariz)
3. Doğru ifadeler de ezbere değil, kavramayı ölçmeli
4. Transkriptte bozuk/hatalı görünen terimler varsa o konudan ifade üretme
5. Türkçe yaz

ZORUNLU ALANLAR — Her ifadede şunlar mutlaka olmalı:
- "ifade": Değerlendirilecek cümle
- "konu": İfadenin ait olduğu ders alt başlığı
- "dogru_mu": true veya false (boolean)
- "aciklama": Neden doğru veya yanlış olduğunun kısa açıklaması

ÇIKTI FORMATI — Sadece bu JSON array'i döndür, başka hiçbir şey yazma:
[
  {{
    "ifade": "...",
    "konu": "...",
    "dogru_mu": true,
    "aciklama": "..."
  }}
]"""

    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    try:
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        sorular = json.loads(text)

        # Kalite kontrolü: eksik alan olanları filtrele
        required_keys = {"ifade", "konu", "dogru_mu", "aciklama"}
        sorular = [s for s in sorular if required_keys.issubset(s.keys())]

        # dogru_mu boolean mı kontrol et (model bazen "true" string döndürüyor)
        for s in sorular:
            if isinstance(s["dogru_mu"], str):
                s["dogru_mu"] = s["dogru_mu"].lower() == "true"

        # Dağılım kontrolü — log'a yaz
        gercek_dogru = sum(1 for s in sorular if s["dogru_mu"])
        gercek_yanlis = len(sorular) - gercek_dogru
        print(f"{len(sorular)} ifade üretildi: {gercek_dogru} doğru, {gercek_yanlis} yanlış")

        return sorular

    except json.JSONDecodeError as e:
        print(f"JSON parse hatası: {e}")
        print("Ham cevap:", response.content[:500])
        return []