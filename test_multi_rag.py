from modules.transcription import load_transcript, split_transcript
from modules.rag import create_vectorstore, ask_question
import shutil
import os

# Eski vektör veritabanını temizle
if os.path.exists("data/vectorstore"):
    shutil.rmtree("data/vectorstore")
    print("Eski vektör veritabanı temizlendi.")

# Tüm transkriptleri yükle ve birleştir
print("\nTranskriptler yükleniyor...")
all_chunks = []

transkriptler = ["data/transcripts/ders1.txt", "data/transcripts/ders2.txt"]

for dosya in transkriptler:
    documents = load_transcript(dosya)
    chunks = split_transcript(documents)
    all_chunks.extend(chunks)
    print(f"{dosya} yüklendi.")

print(f"\nToplam {len(all_chunks)} parça oluşturuldu.")

# Vektör veritabanı oluştur
print("\nVektör veritabanı oluşturuluyor...")
vectorstore = create_vectorstore(all_chunks)

# Farklı derslerden sorular sor
sorular = [
    "Lineer regresyon nedir?",
    "Karar ağaçları nasıl çalışır?",
    "Random Forest nedir?",
    "Gradient descent ne işe yarar?"
]

print("\n--- SORULAR VE CEVAPLAR ---\n")
for soru in sorular:
    cevap = ask_question(soru, vectorstore)
    print(f"Soru: {soru}")
    print(f"Cevap: {cevap}")
    print("-" * 50)