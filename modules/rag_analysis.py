from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


def _get_llm(temperature: float = 0.3):
    """Ortak LLM instance."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


def cevap_analiz_et(vectorstore, soru: dict, ogrenci_cevap: str, llm=None) -> dict:
    """
    Öğrencinin bir soruya verdiği cevabı RAG ile analiz eder.
    llm parametresi verilirse yeni instance yaratmaz — tum_cevaplari_analiz_et'ten paylaşılır.
    """
    if llm is None:
        llm = _get_llm()

    konu = soru.get("konu", "Genel")
    dogru_cevap = soru["dogru_cevap"]
    
    # Boş cevap kontrolü
    if not ogrenci_cevap:
        return {
            "konu": konu,
            "dogru_mu": False,
            "ogrenci_cevap": "",
            "dogru_cevap": dogru_cevap,
            "analiz": f"Bu soru boş bırakıldı. Doğru cevap {dogru_cevap}) {soru['secenekler'].get(dogru_cevap, '')}. Konuyu gözden geçirmeni öneririm."
        }

    dogru_mu = (ogrenci_cevap == dogru_cevap)

    # RAG: konu + soru metnini birleştirerek daha isabetli chunk bul
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(f"{konu} {soru['soru']}")
    context = "\n\n".join([doc.page_content for doc in docs])

    secilen_metin = soru["secenekler"].get(ogrenci_cevap, "Bilinmiyor")
    dogru_metin = soru["secenekler"].get(dogru_cevap, "")
    
    durum = "DOĞRU" if dogru_mu else "YANLIŞ"

    system_prompt = """Sen bir eğitim asistanısın. Öğrencinin quiz cevaplarını ders transkriptine dayanarak analiz ediyorsun.
Görevin: Kısa, öğretici ve motive edici geri bildirim vermek.
Ton: Samimi, destekleyici — yargılamadan açıkla."""

    user_prompt = f"""Öğrencinin aşağıdaki soruya verdiği cevabı analiz et.

SORU: {soru['soru']}
KONU: {konu}
ÖĞRENCİNİN CEVABI: {ogrenci_cevap}) {secilen_metin}  →  {durum}
DOĞRU CEVAP: {dogru_cevap}) {dogru_metin}

DERS TRANSKRİPTİNDEN İLGİLİ BÖLÜM:
{context}

{"YANLIŞ cevap için yazım kuralları:" if not dogru_mu else "DOĞRU cevap için yazım kuralları:"}
{"- Öğrencinin hangi kavramı karıştırmış olabileceğini açıkla" if not dogru_mu else "- Kısa bir pekiştirme yap"}
{"- Doğru cevabın neden doğru olduğunu transkripte dayanarak göster" if not dogru_mu else "- Konunun neden önemli olduğunu 1 cümleyle hatırlat"}
{"- Hocanın bu konuyu nasıl anlattığını 1 cümleyle hatırlat (transkriptte varsa)" if not dogru_mu else ""}
- Türkçe yaz
- Maksimum 60 kelime — özlü ve net ol
- Transkriptte bu konu yoksa genel bilgine dayanarak açıkla"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "konu": konu,
        "dogru_mu": dogru_mu,
        "ogrenci_cevap": ogrenci_cevap,
        "dogru_cevap": dogru_cevap,
        "analiz": response.content.strip()
    }


def tum_cevaplari_analiz_et(vectorstore, quiz: list, cevaplar: dict) -> list:
    """
    Quizdeki tüm cevapları RAG ile analiz eder.
    LLM instance'ı bir kez yaratıp tüm sorularda paylaşır.
    """
    llm = _get_llm()  # Tek instance, tüm sorularda paylaşılır
    sonuclar = []

    for i, soru in enumerate(quiz):
        ogrenci_cevap = cevaplar.get(i, "")
        try:
            analiz = cevap_analiz_et(vectorstore, soru, ogrenci_cevap, llm=llm)
        except Exception as e:
            # Tek soru hata verse bile diğerleri etkilenmesin
            print(f"Soru {i+1} analiz hatası: {e}")
            analiz = {
                "konu": soru.get("konu", "Genel"),
                "dogru_mu": cevaplar.get(i) == soru.get("dogru_cevap"),
                "ogrenci_cevap": ogrenci_cevap,
                "dogru_cevap": soru.get("dogru_cevap", ""),
                "analiz": "Bu soru için analiz üretilemedi."
            }
        analiz["soru_no"] = i + 1
        analiz["soru"] = soru["soru"]
        sonuclar.append(analiz)

    return sonuclar