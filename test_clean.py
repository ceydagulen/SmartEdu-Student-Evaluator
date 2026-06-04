from modules.transcription import clean_transcript

# Ham transkripti oku
with open("data/transcripts/youtube_transkript.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("HAM METİN (ilk 300 karakter):")
print(raw_text[:300])

print("\n--- TEMİZLENMİŞ METİN ---\n")

clean_text = clean_transcript(raw_text)
print(clean_text[:300])

# Temiz metni kaydet
with open("data/transcripts/youtube_temiz.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("\nTemiz transkript kaydedildi!")