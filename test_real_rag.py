from modules.transcription import load_transcript, split_transcript
from modules.rag import create_vectorstore, ask_question
import shutil
import os

# Eski vektör veritabanını temizle
if os.path.exists("data/vectorstore"):
    shutil.rmtree("data/vectorstore")

# Gerçek transkripti yükle
print("Gerçek transkript yükleniyor...")
documents = load_transcript("data/transcripts/youtube_temiz.txt")
chunks = split_transcript(documents)
print(f"Toplam {len(chunks)} parça oluşturuldu.")

# Vektör veritabanı oluştur
print("\nVektör veritabanı oluşturuluyor...")
vectorstore = create_vectorstore(chunks)

# Sorular sor
sorular = [
    "Bu derste hangi konular işlendi?",
    "Hoca hangi materyalleri kullandı?",
    "Öğrencilere ne tavsiye edildi?",
]

print("\n--- SORULAR VE CEVAPLAR ---\n")
for soru in sorular:
    cevap = ask_question(soru, vectorstore)
    print(f"Soru: {soru}")
    print(f"Cevap: {cevap}")
    print("-" * 50)