from modules.rag import load_vectorstore
from modules.concept_map import extract_concept_map, print_concept_map

# Daha önce oluşturduğumuz vektör veritabanını yükle
print("Vektör veritabanı yükleniyor...")
vectorstore = load_vectorstore("data/vectorstore/nyp_1_02")

# Kavram haritası çıkar
print("Kavram haritası çıkarılıyor...")
concept_map = extract_concept_map(vectorstore)

# Ekrana yazdır
print_concept_map(concept_map)