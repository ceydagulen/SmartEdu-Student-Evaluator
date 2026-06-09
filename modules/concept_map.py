from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

load_dotenv()

def extract_concept_map(vectorstore) -> dict:
    """
    Transkriptten kavram haritası çıkarır.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.3,   # Kavram haritası için tutarlılık önemli, düşük tutuyoruz
        max_tokens=4096,
    )

    # Farklı açılardan chunk çek — dersin tamamını temsil etsin
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    queries = [
        "dersin ana konuları ve temel kavramlar",
        "tanımlar ve açıklamalar",
        "yöntemler teknikler ve algoritmalar",
        "örnekler ve uygulamalar",
    ]
    seen = set()
    all_docs = []
    for q in queries:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    context = "\n\n".join([doc.page_content for doc in all_docs])

    system_prompt = """Sen bir eğitim içeriği analistisisin. Ders transkriptlerini analiz ederek yapılandırılmış kavram haritaları oluşturuyorsun.
Görevin: Öğrencilerin dersin içeriğini bir bakışta anlayabileceği net ve hiyerarşik bir kavram haritası üretmek.
Kural: SADECE verilen JSON formatında cevap ver. Hiçbir ek açıklama veya yorum yazma."""

    user_prompt = f"""Aşağıdaki ders transkriptini analiz ederek kavram haritası çıkar.

TRANSKRIPT:
{context}

KURALLAR:
1. 3 ile 6 arasında ana konu belirle — çok fazla veya az olmasın
2. Her ana konunun 2-4 alt konusu olmalı
3. Alt konular somut ve öğrencinin not tutabileceği kadar açıklayıcı olmalı
4. "anahtar_kavramlar" alanına dersin en kritik terimlerini yaz (quiz soruları bu terimlerden üretilecek)
5. Transkriptte bozuk veya anlamsız terimler varsa kavram haritasına ekleme
6. Tüm alanları Türkçe yaz

ÇIKTI FORMATI — Sadece bu JSON'u döndür, başka hiçbir şey yazma:
{{
  "ders_konusu": "Dersin genel başlığı",
  "ozet": "Dersin 2-3 cümlelik özeti",
  "anahtar_kavramlar": ["kavram1", "kavram2", "kavram3"],
  "ana_konular": [
    {{
      "baslik": "Ana konu başlığı",
      "aciklama": "Bu konunun 1-2 cümlelik açıklaması",
      "alt_konular": [
        {{
          "baslik": "Alt konu başlığı",
          "aciklama": "Kısa açıklama"
        }}
      ]
    }}
  ]
}}"""

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

        concept_map = json.loads(text)

        # Zorunlu alan kontrolü
        if "ana_konular" not in concept_map or not concept_map["ana_konular"]:
            print("Kavram haritası boş döndü.")
            return {}

        print(f"Kavram haritası üretildi: {len(concept_map['ana_konular'])} ana konu")
        return concept_map

    except json.JSONDecodeError as e:
        print(f"JSON parse hatası: {e}")
        print("Ham cevap:", response.content[:500])
        return {}


def print_concept_map(concept_map: dict):
    """
    Kavram haritasını terminalde gösterir.
    """
    if not concept_map:
        print("Kavram haritası boş!")
        return

    print(f"\n📚 DERS KONUSU: {concept_map.get('ders_konusu', 'Bilinmiyor')}")
    print("=" * 50)

    for i, ana_konu in enumerate(concept_map.get("ana_konular", []), 1):
        print(f"\n{i}. {ana_konu['baslik']}")
        print(f"   📝 {ana_konu['aciklama']}")

        for alt in ana_konu.get("alt_konular", []):
            print(f"   ├── {alt['baslik']}")
            print(f"   │   {alt['aciklama']}")