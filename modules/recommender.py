from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from modules.analysis import konu_bazli_analiz, zayif_konular, genel_ozet
import os
import json

load_dotenv()


def _get_llm(temperature: float = 0.7):
    """Ortak LLM instance üretici — her fonksiyon buradan alır."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=2048,
    )


def kisisel_oneri_uret(ogrenci_id: int, ogrenci_adi: str = "Öğrenci") -> str:
    """
    Öğrencinin performans analizine göre kişiselleştirilmiş öneri üretir.
    """
    analiz = konu_bazli_analiz(ogrenci_id)
    zayif = zayif_konular(ogrenci_id)
    ozet = genel_ozet(ogrenci_id)

    if not analiz:
        return "Henüz yeterli veri yok. Birkaç quiz çözünce kişisel önerilerini görebilirsin! 🎯"

    konu_detay = "\n".join([
        f"- {konu}: %{veri['basari_orani']} başarı ({veri['dogru']}/{veri['toplam']} doğru)"
        for konu, veri in analiz.items()
    ])

    # Güçlü konuları da çıkar (>= %80)
    guclu_konular = [k for k, v in analiz.items() if v['basari_orani'] >= 80]

    system_prompt = """Sen motive edici ve yapıcı bir eğitim koçusun. Öğrencinin quiz performans verilerini analiz ederek kişiselleştirilmiş öğrenme önerileri sunuyorsun.
Görevin: Öğrenciyi cesaretlendirirken gelişim alanlarını da net ve nazikçe belirtmek.
Ton: Sıcak, destekleyici, somut — genel laflar değil, veriye dayalı öneriler ver."""

    user_prompt = f"""Aşağıdaki performans verilerine dayanarak {ogrenci_adi} için kişiselleştirilmiş bir öğrenme önerisi yaz.

PERFORMANS VERİLERİ:
- Genel ortalama: %{ozet['ortalama_puan']}
- Toplam deneme: {ozet['deneme_sayisi']}
- Güçlü konular: {', '.join(guclu_konular) if guclu_konular else 'Henüz belirlenmedi'}
- Zayıf konular: {', '.join(zayif) if zayif else 'Yok — tüm konularda başarılı!'}

KONU BAZLI DETAY:
{konu_detay}

YAZIM KURALLARI:
1. Türkçe yaz
2. Önce güçlü yönleri takdir et (varsa)
3. Zayıf konuları nazikçe belirt ve her biri için 1 somut çalışma önerisi ver
4. Motive edici bir cümleyle bitir
5. Maksimum 150 kelime — özlü ve net ol
6. Madde işaretleri kullanabilirsin"""

    llm = _get_llm(temperature=0.7)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content


def ders_onerisi_uret(ogrenci_id: int) -> dict:
    """
    Öğrencinin zayıf konularına göre detaylı öğrenme önerileri üretir.
    """
    analiz = konu_bazli_analiz(ogrenci_id)
    zayif = zayif_konular(ogrenci_id)

    if not zayif:
        return {
            "durum": "basarili",
            "mesaj": "🎉 Tüm konularda başarılısın! Yeni konulara geçebilirsin.",
            "oneriler": []
        }

    zayif_detay = "\n".join([
        f"- {konu}: %{analiz[konu]['basari_orani']} başarı ({analiz[konu]['dogru']}/{analiz[konu]['toplam']} doğru)"
        for konu in zayif
        if konu in analiz
    ])

    # Mesajı veriye göre dinamik yap
    en_zayif = min(zayif, key=lambda k: analiz[k]['basari_orani'] if k in analiz else 100)
    en_zayif_puan = analiz[en_zayif]['basari_orani'] if en_zayif in analiz else 0
    mesaj = f"📚 {len(zayif)} konuda gelişim alanın var. En çok dikkat gereken konu: {en_zayif} (%{en_zayif_puan})"

    system_prompt = """Sen bir eğitim danışmanısın. Öğrencinin zayıf olduğu konular için somut, uygulanabilir öğrenme önerileri üretiyorsun.
Kural: SADECE verilen JSON formatında cevap ver. Hiçbir ek açıklama veya yorum yazma."""

    user_prompt = f"""Aşağıdaki konularda öğrencinin başarısı düşük. Her konu için ayrıntılı bir öğrenme önerisi hazırla.

ZAYIF KONULAR:
{zayif_detay}

ZORUNLU ALANLAR — Her öneri için:
- "konu": Konu adı (aynen yukarıdaki gibi yaz)
- "onem": Bu konunun neden önemli olduğu (1-2 cümle, somut)
- "calisma_yontemi": Nasıl çalışılması gerektiği (adım adım, somut tavsiye)
- "kaynak": Önerilen kaynak türü (örn: "YouTube'da görsel anlatım videoları", "Alıştırma problemleri çöz", "Ders notlarını tekrar oku")

ÇIKTI FORMATI — Sadece bu JSON array'i döndür, başka hiçbir şey yazma:
[
  {{
    "konu": "...",
    "onem": "...",
    "calisma_yontemi": "...",
    "kaynak": "..."
  }}
]"""

    llm = _get_llm(temperature=0.5)  # Öneri için biraz daha tutarlı olsun
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

        oneriler = json.loads(text)

        # Kalite kontrolü
        required_keys = {"konu", "onem", "calisma_yontemi", "kaynak"}
        oneriler = [o for o in oneriler if required_keys.issubset(o.keys())]

        return {
            "durum": "gelisim_gerekli",
            "mesaj": mesaj,
            "oneriler": oneriler
        }

    except json.JSONDecodeError as e:
        print(f"JSON parse hatası: {e}")
        print("Ham cevap:", response.content[:500])
        return {
            "durum": "hata",
            "mesaj": "Öneri üretilemedi.",
            "oneriler": []
        }