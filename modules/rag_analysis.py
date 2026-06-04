from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


def cevap_analiz_et(vectorstore, soru: dict, ogrenci_cevap: str) -> dict:
    """
    Öğrencinin bir soruya verdiği cevabı RAG ile analiz eder.
    Transkripte bakarak bağlamsal açıklama üretir.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    konu = soru.get("konu", "Genel")
    dogru_cevap = soru["dogru_cevap"]
    dogru_mu = (ogrenci_cevap == dogru_cevap)

    # RAG: bu sorunun konusuyla ilgili transkript bölümlerini bul
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(f"{konu} {soru['soru']}")
    context = "\n".join([doc.page_content for doc in docs])

    # Öğrencinin seçtiği şıkkın metni
    secilen_metin = soru["secenekler"].get(ogrenci_cevap, "Boş bırakıldı")
    dogru_metin = soru["secenekler"].get(dogru_cevap, "")

    prompt = f"""Sen bir eğitim asistanısın. Bir öğrencinin quiz cevabını ders transkriptine dayanarak analiz et.

Ders transkriptinden ilgili bölüm:
{context}

Soru: {soru['soru']}
Konu: {konu}
Öğrencinin cevabı: {ogrenci_cevap}) {secilen_metin}
Doğru cevap: {dogru_cevap}) {dogru_metin}
Öğrenci {"DOĞRU" if dogru_mu else "YANLIŞ"} cevap verdi.

Görevin:
- Eğer yanlışsa: Öğrencinin neyi yanlış anlamış olabileceğini transkripte dayanarak açıkla. Hocanın bu konuyu nasıl anlattığını hatırlat.
- Eğer doğruysa: Kısa bir pekiştirme yap, konuyu özetle.
- Türkçe, samimi ve öğretici bir dille yaz.
- Maksimum 80 kelime.
- Transkriptte bu konu yoksa, genel bilgine dayanarak açıkla ama bunu belirt."""

    response = llm.invoke(prompt)

    return {
        "konu": konu,
        "dogru_mu": dogru_mu,
        "ogrenci_cevap": ogrenci_cevap,
        "dogru_cevap": dogru_cevap,
        "analiz": response.content
    }


def tum_cevaplari_analiz_et(vectorstore, quiz: list, cevaplar: dict) -> list:
    """
    Quizdeki tüm cevapları RAG ile analiz eder.
    """
    sonuclar = []
    for i, soru in enumerate(quiz):
        ogrenci_cevap = cevaplar.get(i, "")
        analiz = cevap_analiz_et(vectorstore, soru, ogrenci_cevap)
        analiz["soru_no"] = i + 1
        analiz["soru"] = soru["soru"]
        sonuclar.append(analiz)
    return sonuclar