from langchain_core.documents import Document
import os 
import subprocess
import re

def load_transcript(file_path: str) -> list[Document]:

    """
    Transkript dosyasını okur ve LangChain Document listesine çevirir.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Boş satırları temizle
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # LangChain Document formatına çevir
    document = Document(
        page_content=text,
        metadata={"source": file_path}
    )

    return [document]


def split_transcript(documents: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
    """
    Uzun transkripti küçük parçalara böler (chunking).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    print(f"Transkript {len(chunks)} parçaya bölündü.")
    return chunks

def download_youtube_audio(url: str, output_dir: str = "data/audio") -> str:
    """
    YouTube videosundan ses dosyasını indirir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    command = [
        "yt-dlp",
        "-x",                          # sadece ses
        "--audio-format", "mp3",       # mp3 formatında
        "--audio-quality", "0",        # en iyi kalite
        "-o", output_path,
        url
    ]
    
    print("Video indiriliyor...")
    subprocess.run(command, check=True)
    
    # İndirilen dosyayı bul
    for file in os.listdir(output_dir):
        if file.endswith(".mp3"):
            return os.path.join(output_dir, file)
    
    raise FileNotFoundError("Ses dosyası indirilemedi.")


def transcribe_audio(audio_path: str) -> str:
    """
    Whisper ile ses dosyasını metne çevirir.
    """
    import whisper
    
    print(f"Transkript oluşturuluyor: {audio_path}")
    print("Bu işlem birkaç dakika sürebilir...")
    
    model = whisper.load_model("medium")  # base, small, medium, large
    result = model.transcribe(audio_path, language="tr")
    
    return result["text"]


def youtube_to_transcript(url: str, output_file: str = None) -> str:
    """
    YouTube URL'sinden transkript oluşturur ve dosyaya kaydeder.
    """
    # Sesi indir
    audio_path = download_youtube_audio(url)
    
    # Transkripte çevir
    transcript = transcribe_audio(audio_path)
    
    # Dosyaya kaydet
    if output_file is None:
        output_file = "data/transcripts/youtube_transkript.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print(f"Transkript kaydedildi: {output_file}")
    return transcript

def clean_transcript(text: str) -> str:
    """
    Ham transkripti temizler ve düzenler.
    """
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Dolgu kelimeleri temizle
    doldurucular = [
        r'\beee+\b', r'\bşey\b', r'\byani\b', r'\bımmm*\b',
        r'\bhmm+\b', r'\behh*\b', r'\baa+\b', r'\böö+\b'
    ]
    for dolgu in doldurucular:
        text = re.sub(dolgu, '', text, flags=re.IGNORECASE)
    
    # Tekrarlayan kelimeleri temizle (yani yani yani → yani)
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text, flags=re.IGNORECASE)
    
    # Fazla boşlukları tekrar temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Cümleleri satırlara böl
    text = re.sub(r'(?<=[.!?])\s+', '\n', text)
    
    return text


def process_youtube_transcript(url: str, output_file: str = None) -> list:
    """
    YouTube videosunu indirir, transkripte çevirir, temizler ve
    LangChain Document listesi olarak döndürür.
    """
    # Transkripti al
    raw_transcript = youtube_to_transcript(url, output_file)
    
    # Temizle
    print("Transkript temizleniyor...")
    clean_text = clean_transcript(raw_transcript)
    
    # Temiz transkripti kaydet
    clean_file = output_file.replace('.txt', '_temiz.txt') if output_file else "data/transcripts/youtube_temiz.txt"
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write(clean_text)
    
    print(f"Temiz transkript kaydedildi: {clean_file}")
    
    # LangChain Document formatına çevir
    documents = load_transcript(clean_file)
    chunks = split_transcript(documents)
    
    return chunks