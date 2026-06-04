import sqlite3
import os
from datetime import datetime

DB_PATH = "data/smartedu.db"


def init_db():
    """
    Veritabanını ve tabloları oluşturur.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Öğrenciler tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ogrenciler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ad TEXT NOT NULL,
            ogrenci_no TEXT UNIQUE
        )
    """)

    # Quiz denemeleri tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS denemeler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ogrenci_id INTEGER,
            ders_adi TEXT,
            tarih TEXT,
            puan REAL,
            dogru_sayisi INTEGER,
            toplam_soru INTEGER,
            FOREIGN KEY (ogrenci_id) REFERENCES ogrenciler (id)
        )
    """)

    # Cevap detayları tablosu (konu bazlı analiz için kritik)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cevaplar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deneme_id INTEGER,
            soru TEXT,
            konu TEXT,
            ogrenci_cevap TEXT,
            dogru_cevap TEXT,
            dogru_mu INTEGER,
            FOREIGN KEY (deneme_id) REFERENCES denemeler (id)
        )
    """)

    conn.commit()
    conn.close()
    print("Veritabanı hazır.")


def ogrenci_ekle_veya_getir(ad: str, ogrenci_no: str) -> int:
    """
    Öğrenci varsa id'sini döndürür, yoksa ekler.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM ogrenciler WHERE ogrenci_no = ?", (ogrenci_no,))
    sonuc = cursor.fetchone()

    if sonuc:
        ogrenci_id = sonuc[0]
    else:
        cursor.execute(
            "INSERT INTO ogrenciler (ad, ogrenci_no) VALUES (?, ?)",
            (ad, ogrenci_no)
        )
        ogrenci_id = cursor.lastrowid
        conn.commit()

    conn.close()
    return ogrenci_id


def deneme_kaydet(ogrenci_id: int, ders_adi: str, quiz: list, cevaplar: dict) -> int:
    """
    Quiz denemesini ve tüm cevap detaylarını kaydeder.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Puanı hesapla
    dogru_sayisi = 0
    for i, soru in enumerate(quiz):
        if cevaplar.get(i) == soru["dogru_cevap"]:
            dogru_sayisi += 1

    toplam = len(quiz)
    puan = (dogru_sayisi / toplam) * 100 if toplam > 0 else 0

    # Denemeyi kaydet
    cursor.execute("""
        INSERT INTO denemeler (ogrenci_id, ders_adi, tarih, puan, dogru_sayisi, toplam_soru)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ogrenci_id, ders_adi, datetime.now().strftime("%Y-%m-%d %H:%M"), puan, dogru_sayisi, toplam))

    deneme_id = cursor.lastrowid

    # Her cevabı detaylı kaydet
    for i, soru in enumerate(quiz):
        ogrenci_cevap = cevaplar.get(i, "")
        dogru_cevap = soru["dogru_cevap"]
        dogru_mu = 1 if ogrenci_cevap == dogru_cevap else 0
        konu = soru.get("konu", "Genel")

        cursor.execute("""
            INSERT INTO cevaplar (deneme_id, soru, konu, ogrenci_cevap, dogru_cevap, dogru_mu)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (deneme_id, soru["soru"], konu, ogrenci_cevap, dogru_cevap, dogru_mu))

    conn.commit()
    conn.close()
    return deneme_id