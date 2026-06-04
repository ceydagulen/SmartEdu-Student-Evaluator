from modules.recommender import kisisel_oneri_uret, ders_onerisi_uret

# Daha önce kaydedilen Ceyda öğrencisi (id=1)
ogrenci_id = 1

print("=== KİŞİSEL ÖNERİ ===\n")
print(kisisel_oneri_uret(ogrenci_id, "Ceyda"))

print("\n\n=== ÖNERİLEN DERSLER/KAYNAKLAR ===\n")
sonuc = ders_onerisi_uret(ogrenci_id)
print(f"Durum: {sonuc['mesaj']}\n")

for oneri in sonuc["oneriler"]:
    print(f"📚 {oneri['konu']}")
    print(f"   Önem: {oneri['onem']}")
    print(f"   Çalışma yöntemi: {oneri['calisma_yontemi']}")
    print(f"   Kaynak: {oneri['kaynak']}")
    print()