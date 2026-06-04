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
        model="llama-3.3-70b-versatile"
    )

    # RAG ile dersin içeriğini al
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    docs = retriever.invoke("dersin tüm konuları ana başlıklar alt başlıklar")
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Aşağıdaki ders transkriptini analiz et ve kavram haritası çıkar.

Transkript:
{context}

Kurallar:
- Ana konuları ve alt konuları belirle
- Konular arasındaki ilişkileri göster
- Sadece JSON formatında döndür, başka hiçbir şey yazma

JSON formatı:
{{
  "ders_konusu": "Dersin genel konusu",
  "ana_konular": [
    {{
      "baslik": "Ana konu başlığı",
      "aciklama": "Kısa açıklama",
      "alt_konular": [
        {{
          "baslik": "Alt konu",
          "aciklama": "Kısa açıklama"
        }}
      ]
    }}
  ]
}}"""

    response = llm.invoke(prompt)

    try:
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        concept_map = json.loads(text)
        return concept_map
    except json.JSONDecodeError:
        print("JSON parse hatası:")
        print(response.content)
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