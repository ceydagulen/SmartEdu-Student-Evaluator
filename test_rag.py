from modules.transcription import load_transcript, split_transcript
from modules.rag import create_vectorstore, ask_question

# 1. Transkripti yükle ve parçala
print("Transkript yükleniyor...")
documents = load_transcript("data/transcripts/ders1.txt")
chunks = split_transcript(documents)

# 2. Vektör veritabanı oluştur
print("\nVektör veritabanı oluşturuluyor...")
vectorstore = create_vectorstore(chunks)

# 3. Soru sor
print("\nSoru soruluyor...")
soru = "Lineer regresyon nedir ve nasıl çalışır?"
cevap = ask_question(soru, vectorstore)

print(f"\nSoru: {soru}")
print(f"\nCevap: {cevap}")