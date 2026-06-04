from langchain_groq import ChatGroq
from dotenv import load_dotenv
from modules.analysis import konu_bazli_analiz, zayif_konular, genel_ozet
import os

load_dotenv()


def kisisel_oneri_uret(ogrenci_id: int, ogrenci_adi: str = "Öğrenci") -> str:
    """
    Öğrencinin performans analizine göre kişiselleştirilmiş öneri üretir.
    """
    # Analiz verilerini topla
    analiz = konu_bazli_analiz(ogrenci_id)
    zayif = zayif_konular(ogrenci_id)
    ozet = genel_ozet(ogrenci_id)

    if not analiz:
        return "Henüz yeterli veri yok. Birkaç quiz çözünce kişisel önerilerini görebilirsin!"

    # Analiz verisini metne çevir
    konu_detay = "\n".join([
        f"- {konu}: %{veri['basari_orani']} başarı ({veri['dogru']}/{veri['toplam']} doğru)"
        for konu, veri in analiz.items()
    ])

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    prompt = f"""Sen bir eğitim danışmanısın. Aşağıda bir öğrencinin quiz performans analizi var.
Bu verilere dayanarak öğrenciye kişiselleştirilmiş, motive edici ve yapıcı bir öğrenme önerisi yaz.

Öğrenci: {ogrenci_adi}
Genel ortalama: %{ozet['ortalama_puan']}
Toplam deneme: {ozet['deneme_sayisi']}

Konu bazlı performans:
{konu_detay}

Zayıf olduğu konular: {', '.join(zayif) if zayif else 'Yok, tüm konularda başarılı!'}

Kurallar:
- Türkçe yaz
- Önce güçlü yönlerini takdir et
- Sonra zayıf konuları nazikçe belirt
- Her zayıf konu için somut çalışma önerisi ver
- Motive edici bir dille bitir
- Maksimum 200 kelime
- Madde işaretleri kullanabilirsin"""

    response = llm.invoke(prompt)
    return response.content

def ders_onerisi_uret(ogrenci_id: int, vectorstore=None) -> dict:
    """
    Öğrencinin zayıf konularına göre tekrar edilecek konular ve
    önerilen kaynaklar sunar.
    """
    from modules.analysis import konu_bazli_analiz, zayif_konular

    analiz = konu_bazli_analiz(ogrenci_id)
    zayif = zayif_konular(ogrenci_id)

    if not zayif:
        return {
            "durum": "basarili",
            "mesaj": "Tüm konularda başarılısın! Yeni konulara geçebilirsin.",
            "oneriler": []
        }

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    zayif_detay = "\n".join([
        f"- {konu}: %{analiz[konu]['basari_orani']} başarı"
        for konu in zayif
    ])

    prompt = f"""Bir öğrencinin quiz analizine göre zayıf olduğu konular şunlar:

{zayif_detay}

Her zayıf konu için bir öğrenme önerisi hazırla. Her öneri şunları içermeli:
- Konu adı
- Neden önemli olduğu (1 cümle)
- Nasıl çalışılması gerektiği (somut tavsiye)
- Önerilen kaynak türü (video, kitap bölümü, alıştırma vb.)

Sadece JSON formatında döndür, başka hiçbir şey yazma:
[
  {{
    "konu": "Konu adı",
    "onem": "Neden önemli",
    "calisma_yontemi": "Nasıl çalışmalı",
    "kaynak": "Önerilen kaynak türü"
  }}
]"""

    response = llm.invoke(prompt)

    import json
    try:
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        oneriler = json.loads(text)
        return {
            "durum": "gelisim_gerekli",
            "mesaj": f"{len(zayif)} konuda gelişim alanın var.",
            "oneriler": oneriler
        }
    except json.JSONDecodeError:
        return {
            "durum": "hata",
            "mesaj": "Öneri üretilemedi.",
            "oneriler": []
        }