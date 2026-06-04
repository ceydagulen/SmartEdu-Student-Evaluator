from modules.transcription import youtube_to_transcript

url = "https://www.youtube.com/watch?v=QxB4My0QIKc"

print("YouTube'dan transkript oluşturuluyor...")
transcript = youtube_to_transcript(url)

print("\n--- TRANSKRİPT ÖNIZLEME ---")
print(transcript[:500])
print("...")
print(f"\nToplam karakter sayısı: {len(transcript)}")