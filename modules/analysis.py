import sqlite3

DB_PATH = "data/smartedu.db"


def konu_bazli_analiz(ogrenci_id: int) -> dict:
    """
    Öğrencinin konu bazlı performansını hesaplar.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.konu,
               COUNT(*) as toplam,
               SUM(c.dogru_mu) as dogru
        FROM cevaplar c
        JOIN denemeler d ON c.deneme_id = d.id
        WHERE d.ogrenci_id = ?
        GROUP BY c.konu
    """, (ogrenci_id,))

    sonuclar = cursor.fetchall()
    conn.close()

    analiz = {}
    for konu, toplam, dogru in sonuclar:
        basari = (dogru / toplam) * 100 if toplam > 0 else 0
        analiz[konu] = {
            "toplam": toplam,
            "dogru": dogru,
            "basari_orani": round(basari, 1)
        }

    return analiz


def zayif_konular(ogrenci_id: int, esik: float = 60.0) -> list:
    """
    Başarı oranı eşiğin altında olan konuları döndürür.
    """
    analiz = konu_bazli_analiz(ogrenci_id)
    zayif = [
        konu for konu, veri in analiz.items()
        if veri["basari_orani"] < esik
    ]
    return zayif


def gelisim_takibi(ogrenci_id: int) -> list:
    """
    Öğrencinin zaman içindeki quiz puanlarını döndürür.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ders_adi, tarih, puan
        FROM denemeler
        WHERE ogrenci_id = ?
        ORDER BY tarih
    """, (ogrenci_id,))

    sonuclar = cursor.fetchall()
    conn.close()

    return [
        {"ders": ders, "tarih": tarih, "puan": puan}
        for ders, tarih, puan in sonuclar
    ]


def genel_ozet(ogrenci_id: int) -> dict:
    """
    Öğrencinin genel performans özetini döndürür.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) as deneme_sayisi,
               AVG(puan) as ortalama_puan,
               MAX(puan) as en_yuksek,
               MIN(puan) as en_dusuk
        FROM denemeler
        WHERE ogrenci_id = ?
    """, (ogrenci_id,))

    sonuc = cursor.fetchone()
    conn.close()

    if sonuc and sonuc[0] > 0:
        return {
            "deneme_sayisi": sonuc[0],
            "ortalama_puan": round(sonuc[1], 1),
            "en_yuksek": round(sonuc[2], 1),
            "en_dusuk": round(sonuc[3], 1)
        }
    return {"deneme_sayisi": 0, "ortalama_puan": 0, "en_yuksek": 0, "en_dusuk": 0}