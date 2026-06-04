from modules.transcription import load_transcript, split_transcript
from modules.rag import load_vectorstore, create_vectorstore
from modules.quiz import generate_quiz, print_quiz
import os
import shutil

# Vektör veritabanını yükle
print("Vektör veritabanı yükleniyor...")

if os.path.exists("data/vectorstore/ders_1"):
    vectorstore = load_vectorstore("data/vectorstore/ders_1")
else:
    # Yoksa oluştur
    print("Veritabanı bulunamadı, oluşturuluyor...")
    shutil.rmtree("data/vectorstore", ignore_errors=True)
    documents = load_transcript("data/transcripts/youtube_temiz.txt")
    chunks = split_transcript(documents)
    vectorstore = create_vectorstore(chunks, "data/vectorstore/ders_1")

# Quiz üret
print("\nQuiz üretiliyor...")
quiz = generate_quiz(vectorstore, ders_id="ders_1", soru_sayisi=5)

# Quizi ekrana yazdır
print_quiz(quiz)