from langchain_core.documents import Document
import os

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