import os
import re
import shutil
import subprocess
from pathlib import Path

import streamlit as st

from modules.transcription import (
    transcribe_audio, clean_transcript, load_transcript, split_transcript,
)
from modules.rag import create_vectorstore, load_vectorstore
from modules.quiz import generate_quiz, generate_true_false
from modules.concept_map import extract_concept_map
from modules.database import init_db, ogrenci_ekle_veya_getir, deneme_kaydet
from modules.analysis import gelisim_takibi, genel_ozet, konu_bazli_analiz
from modules.recommender import kisisel_oneri_uret, ders_onerisi_uret
from modules.rag_analysis import tum_cevaplari_analiz_et


st.set_page_config(page_title="SmartEdu | AI Learning Assistant", page_icon=":mortar_board:", layout="wide")
init_db()


def slugify(text):
    text = text.strip().lower()
    tr = {"ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}
    for a, b in tr.items():
        text = text.replace(a, b)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "ders"


def download_from_url(url, output_dir="data/audio"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "%(title)s.%(ext)s")
    command = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "0", "-o", output_path, url]
    subprocess.run(command, check=True)
    mp3_files = list(Path(output_dir).glob("*.mp3"))
    if not mp3_files:
        raise FileNotFoundError("Ses dosyasi indirilemedi.")
    return str(max(mp3_files, key=lambda f: f.stat().st_mtime))


def reset_course_state():
    st.session_state.quiz = []
    st.session_state.dy_sorular = []
    st.session_state.cevaplar = {}
    st.session_state.dy_cevaplar = {}
    st.session_state.ders_adi = ""
    st.session_state.vectorstore_path = ""
    st.session_state.concept_map = None


def go_to(stage):
    st.session_state.asama = stage
    st.rerun()


def quiz_uret(vectorstore, tip, cs_sayi, dy_sayi):
    quiz, dy_sorular = [], []
    if tip in ("Coktan Secmeli", "Ikisi de"):
        quiz = generate_quiz(vectorstore, soru_sayisi=cs_sayi)
    if tip in ("Dogru/Yanlis", "Ikisi de"):
        try:
            dy_sorular = generate_true_false(vectorstore, soru_sayisi=dy_sayi)
        except Exception:
            dy_sorular = []
    return quiz, dy_sorular


def render_concept_map(cm):
    if not cm or not isinstance(cm, dict) or not cm.get("ana_konular"):
        return
    with st.expander("📚 " + cm.get("ders_konusu", "Ders Kavram Haritası"), expanded=False):
        if cm.get("ozet"):
            st.info(cm["ozet"])
        if cm.get("anahtar_kavramlar"):
            st.markdown("**🔑 Anahtar Kavramlar:**")
            kavramlar = " &nbsp;|&nbsp; ".join([f"`{k}`" for k in cm["anahtar_kavramlar"]])
            st.markdown(kavramlar, unsafe_allow_html=True)
            st.markdown("---")
        for i, ana in enumerate(cm.get("ana_konular", []), 1):
            st.markdown(f"**{i}. {ana.get('baslik', '')}**")
            if ana.get("aciklama"):
                st.caption(ana["aciklama"])
            for alt in ana.get("alt_konular", []):
                aciklama = f" — {alt.get('aciklama', '')}" if alt.get("aciklama") else ""
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;▸ **{alt.get('baslik', '')}**{aciklama}",
                    unsafe_allow_html=True
                )
            if i < len(cm.get("ana_konular", [])):
                st.markdown("---")


def transkript_isle_ve_quiz_olustur(transkript_metni, ders_adi, quiz_tip, cs_sayi, dy_sayi):
    ders_key = slugify(ders_adi)
    progress = st.progress(0, text="İşlem başlatılıyor...")

    clean = clean_transcript(transkript_metni)
    os.makedirs("data/transcripts", exist_ok=True)
    tpath = "data/transcripts/" + ders_key + ".txt"
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(clean)
    progress.progress(25, text="Transkript hazır.")

    try:
        with st.status("İçerik analiz ediliyor...", expanded=True) as status:
            docs = load_transcript(tpath)
            chunks = split_transcript(docs)
            vpath = "data/vectorstore/" + ders_key
            if os.path.exists(vpath):
                shutil.rmtree(vpath)
            vectorstore = create_vectorstore(chunks, vpath)
            status.update(label="Analiz tamamlandı.", state="complete")
        progress.progress(55, text="Analiz tamamlandı.")
    except Exception as e:
        st.error("Analiz hatası: " + str(e))
        st.stop()

    try:
        with st.status("Kavramlar çıkarılıyor...", expanded=True) as status:
            concept_map = extract_concept_map(vectorstore)
            status.update(label="Kavramlar hazır.", state="complete")
        progress.progress(75, text="Kavramlar çıkarıldı.")
    except Exception:
        concept_map = None

    with st.status("Quiz üretiliyor...", expanded=True) as status:
        quiz, dy_sorular = quiz_uret(vectorstore, quiz_tip, cs_sayi, dy_sayi)
        status.update(label="Quiz hazır.", state="complete")
    progress.progress(100, text="Tamamlandı.")

    return vectorstore, vpath, concept_map, quiz, dy_sorular


def cevaplanan_sayisi_hesapla(toplam_cs, toplam_dy):
    """
    Widget key'lerinden güncel cevap sayısını hesaplar.
    session_state.cevaplar yerine doğrudan radio key'lerine bakar —
    Streamlit'in render gecikmesinden etkilenmez.
    """
    cs = sum(
        1 for i in range(toplam_cs)
        if st.session_state.get(f"radio_{i}") is not None
    )
    dy = sum(
        1 for i in range(toplam_dy)
        if st.session_state.get(f"dy_radio_{i}") is not None
    )
    return cs, dy


# ===== CSS =====
st.markdown("""
<style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    .main-header { padding: 28px 32px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 18px; color: white; margin-bottom: 24px; box-shadow: 0 8px 28px rgba(118, 75, 162, 0.25); }
    .main-header h1 { margin: 0; font-size: 34px; font-weight: 800; }
    .main-header p { margin: 10px 0 0 0; opacity: 0.95; font-size: 16px; max-width: 900px; }
    .info-card { background: #faf9fd; border: 1px solid #e8e3f3; border-radius: 14px; padding: 18px 20px; min-height: 120px; }
    .info-card h4 { margin: 0 0 8px 0; color: #5b3f91; }
    .info-card p { margin: 0; color: #4b4b5c; font-size: 14px; line-height: 1.45; }
    .section-title { font-size: 22px; font-weight: 750; margin: 6px 0 12px 0; }
    .stButton > button { border-radius: 12px; font-weight: 700; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border: none; min-height: 44px; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(118, 75, 162, 0.30); }
    .soru-kart { background: #faf9fd; border: 1px solid #e8e3f3; border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; }
    .soru-no { font-size: 13px; font-weight: 700; color: #764ba2; margin-bottom: 6px; }
    .soru-konu { font-size: 11px; color: #9b8ab8; margin-bottom: 8px; }
    .fb-dogru { background: #e8f5e9; border-left: 5px solid #4caf50; padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; color: #2e7d32; font-weight: 700; }
    .fb-yanlis { background: #fdecea; border-left: 5px solid #ef5350; padding: 12px 16px; border-radius: 10px; margin-bottom: 8px; color: #c62828; font-weight: 700; }
    .fb-aciklama { background: #f3f0fa; border-left: 5px solid #764ba2; padding: 11px 16px; border-radius: 10px; margin-bottom: 16px; color: #4a3b6b; font-size: 14px; line-height: 1.5; }
    [data-testid="stMetric"] { background: #faf9fd; padding: 16px; border-radius: 14px; border: 1px solid #e8e3f3; }
    [data-testid="stMetric"] label { color: #5b3f91 !important; }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>SmartEdu</h1>
    <p>Ders transkriptlerini analiz ederek otomatik quiz oluşturan, öğrenci cevaplarını değerlendiren ve kişiselleştirilmiş öğrenme önerileri sunan yapay zeka destekli eğitim asistanı.</p>
</div>
""", unsafe_allow_html=True)

default_state = {
    "asama": "giris", "quiz": [], "dy_sorular": [], "cevaplar": {}, "dy_cevaplar": {},
    "ders_adi": "", "ogrenci_id": None, "ogrenci_adi": "", "vectorstore_path": "", "concept_map": None,
}
for key, val in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown("## Sistem Akışı")
    st.markdown("1. Öğrenci girişi\n2. Ders yükleme / seçme\n3. Transkript ve analiz\n4. Quiz üretimi\n5. Cevap analizi\n6. Kişisel öneriler")
    st.divider()
    if st.session_state.ogrenci_adi:
        st.success("Öğrenci: " + st.session_state.ogrenci_adi)
    else:
        st.info("Henüz giriş yapılmadı.")
    st.caption("AI / NLP / RAG / ChromaDB / SQLite")


# ===== GİRİŞ =====
if st.session_state.asama == "giris":
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="section-title">Öğrenci Girişi</div>', unsafe_allow_html=True)
        ad = st.text_input("Ad Soyad", placeholder="Örn: Ceyda Gülen")
        ogrenci_no = st.text_input("Öğrenci No", placeholder="Örn: 21360859042")
        if st.button("Giriş Yap", use_container_width=True):
            if ad.strip() and ogrenci_no.strip():
                st.session_state.ogrenci_id = ogrenci_ekle_veya_getir(ad.strip(), ogrenci_no.strip())
                st.session_state.ogrenci_adi = ad.strip()
                reset_course_state()
                go_to("ders_sec")
            else:
                st.error("Lütfen ad soyad ve öğrenci numarasını girin.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="info-card"><h4>Ders Analizi</h4><p>Zoom/Meet/Teams transkriptleri veya video URL\'si ile ders içeriği analiz edilir.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="info-card"><h4>Otomatik Quiz</h4><p>Çoktan seçmeli ve doğru/yanlış soruları otomatik oluşturulur.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="info-card"><h4>Kişisel Öneri</h4><p>Cevaplar analiz edilir ve eksik konulara göre öneriler sunulur.</p></div>', unsafe_allow_html=True)


# ===== DERS SEÇ =====
elif st.session_state.asama == "ders_sec":
    st.markdown("### Hoş geldin, " + st.session_state.ogrenci_adi + "!")
    tab1, tab2 = st.tabs(["Quiz Oluştur", "Performansım"])

    with tab1:
        kaynak = st.radio(
            "Ders kaynağı seç:",
            ["Video URL (YouTube vb.)", "Transkript Yapıştır / Yükle", "Kayıtlı dersten seç"],
            horizontal=True
        )

        st.markdown("**Quiz Ayarları**")
        quiz_tip = st.radio("Quiz tipi:", ["Çoktan Seçmeli", "Doğru/Yanlış", "İkisi de"], horizontal=True)
        cc1, cc2 = st.columns(2)
        cs_sayi = dy_sayi = 0
        if quiz_tip in ("Çoktan Seçmeli", "İkisi de"):
            cs_sayi = cc1.slider("Çoktan seçmeli soru sayısı", 5, 30, 10)
        if quiz_tip in ("Doğru/Yanlış", "İkisi de"):
            dy_sayi = cc2.slider("Doğru/Yanlış soru sayısı", 5, 30, 10)

        tip_map = {"Çoktan Seçmeli": "Coktan Secmeli", "Doğru/Yanlış": "Dogru/Yanlis", "İkisi de": "Ikisi de"}
        quiz_tip_ic = tip_map[quiz_tip]

        st.markdown("---")

        # ===== KAYNAK 1: VIDEO URL =====
        if kaynak == "Video URL (YouTube vb.)":
            ders_adi = st.text_input("Ders Adı", placeholder="Örn: NumPy Temelleri")
            url = st.text_input("Ders Video URL", placeholder="https://www.youtube.com/watch?v=...")
            st.caption("YouTube veya erişilebilir çevrim içi ders video bağlantısı giriniz.")

            if st.button("Analiz Et ve Quiz Oluştur", use_container_width=True):
                if not ders_adi.strip() or not url.strip():
                    st.error("Lütfen ders adını ve video bağlantısını girin.")
                    st.stop()

                ders_key = slugify(ders_adi)
                st.session_state.ders_adi = ders_adi.strip()
                progress = st.progress(0, text="İşlem başlatılıyor...")

                try:
                    with st.status("Video indiriliyor...", expanded=True) as status:
                        if os.path.exists("data/audio"):
                            shutil.rmtree("data/audio")
                        audio_path = download_from_url(url.strip())
                        status.update(label="Video indirildi.", state="complete")
                    progress.progress(15, text="Video indirildi.")
                except Exception as e:
                    st.error("İndirme hatası: " + str(e)); st.stop()

                try:
                    with st.status("Transkript oluşturuluyor...", expanded=True) as status:
                        st.write("Bu işlem birkaç dakika sürebilir.")
                        raw = transcribe_audio(audio_path)
                        clean = clean_transcript(raw)
                        os.makedirs("data/transcripts", exist_ok=True)
                        tpath = "data/transcripts/" + ders_key + ".txt"
                        with open(tpath, "w", encoding="utf-8") as f:
                            f.write(clean)
                        status.update(label="Transkript hazır.", state="complete")
                    progress.progress(40, text="Transkript hazır.")
                except Exception as e:
                    st.error("Transkript hatası: " + str(e)); st.stop()

                try:
                    with st.status("İçerik analiz ediliyor...", expanded=True) as status:
                        docs = load_transcript(tpath)
                        chunks = split_transcript(docs)
                        vpath = "data/vectorstore/" + ders_key
                        if os.path.exists(vpath):
                            shutil.rmtree(vpath)
                        vectorstore = create_vectorstore(chunks, vpath)
                        status.update(label="Analiz tamamlandı.", state="complete")
                    progress.progress(65, text="Analiz tamamlandı.")
                except Exception as e:
                    st.error("Analiz hatası: " + str(e)); st.stop()

                try:
                    with st.status("Kavramlar çıkarılıyor...", expanded=True) as status:
                        concept_map = extract_concept_map(vectorstore)
                        status.update(label="Kavramlar hazır.", state="complete")
                    progress.progress(80, text="Kavramlar çıkarıldı.")
                except Exception:
                    concept_map = None

                with st.status("Quiz üretiliyor...", expanded=True) as status:
                    quiz, dy_sorular = quiz_uret(vectorstore, quiz_tip_ic, cs_sayi, dy_sayi)
                    status.update(label="Quiz hazır.", state="complete")
                progress.progress(100, text="Tamamlandı.")

                st.session_state.quiz = quiz
                st.session_state.dy_sorular = dy_sorular
                st.session_state.cevaplar = {}
                st.session_state.dy_cevaplar = {}
                st.session_state.concept_map = concept_map
                st.session_state.vectorstore_path = vpath
                go_to("quiz")

        # ===== KAYNAK 2: TRANSKRİPT YAPIŞTIR / YÜKLE =====
        elif kaynak == "Transkript Yapıştır / Yükle":
            ders_adi = st.text_input("Ders Adı", placeholder="Örn: Veri Yapıları - Hafta 3")

            giris_tipi = st.radio(
                "Transkript nasıl eklemek istersin?",
                ["Metin olarak yapıştır", ".txt dosyası yükle"],
                horizontal=True
            )

            transkript_metni = ""

            if giris_tipi == "Metin olarak yapıştır":
                transkript_metni = st.text_area(
                    "Transkripti buraya yapıştır",
                    placeholder="Zoom / Google Meet / Microsoft Teams otomatik transkriptini buraya yapıştırabilirsiniz...",
                    height=250
                )
                if transkript_metni:
                    st.caption(f"📄 {len(transkript_metni):,} karakter · yaklaşık {len(transkript_metni.split()):,} kelime")
            else:
                dosya = st.file_uploader(
                    "Transkript dosyası yükle (.txt)",
                    type=["txt"],
                    help="Zoom/Meet/Teams'den dışa aktarılan .txt transkript dosyası"
                )
                if dosya:
                    transkript_metni = dosya.read().decode("utf-8")
                    st.success(f"✅ '{dosya.name}' yüklendi — {len(transkript_metni):,} karakter")

            if st.button("Analiz Et ve Quiz Oluştur", use_container_width=True):
                if not ders_adi.strip():
                    st.error("Lütfen ders adını girin.")
                    st.stop()
                if not transkript_metni.strip():
                    st.error("Lütfen transkript girin veya dosya yükleyin.")
                    st.stop()
                if len(transkript_metni.strip()) < 200:
                    st.error("Transkript çok kısa. Lütfen daha fazla içerik ekleyin.")
                    st.stop()

                st.session_state.ders_adi = ders_adi.strip()
                _, vpath, concept_map, quiz, dy_sorular = transkript_isle_ve_quiz_olustur(
                    transkript_metni, ders_adi.strip(), quiz_tip_ic, cs_sayi, dy_sayi
                )
                st.session_state.quiz = quiz
                st.session_state.dy_sorular = dy_sorular
                st.session_state.cevaplar = {}
                st.session_state.dy_cevaplar = {}
                st.session_state.concept_map = concept_map
                st.session_state.vectorstore_path = vpath
                go_to("quiz")

        # ===== KAYNAK 3: KAYITLI DERS =====
        else:
            kayitli = []
            if os.path.exists("data/vectorstore"):
                kayitli = [d for d in os.listdir("data/vectorstore") if os.path.isdir("data/vectorstore/" + d)]
            if not kayitli:
                st.info("Henüz kayıtlı ders yok. Önce yeni bir ders ekleyin.")
            else:
                secili = st.selectbox("Kayıtlı Ders Seç", kayitli)
                if st.button("Bu Dersten Quiz Oluştur", use_container_width=True):
                    vpath = "data/vectorstore/" + secili
                    st.session_state.ders_adi = secili
                    with st.spinner("Quiz oluşturuluyor..."):
                        vectorstore = load_vectorstore(vpath)
                        quiz, dy_sorular = quiz_uret(vectorstore, quiz_tip_ic, cs_sayi, dy_sayi)
                        try:
                            concept_map = extract_concept_map(vectorstore)
                        except Exception:
                            concept_map = None
                    st.session_state.quiz = quiz
                    st.session_state.dy_sorular = dy_sorular
                    st.session_state.cevaplar = {}
                    st.session_state.dy_cevaplar = {}
                    st.session_state.concept_map = concept_map
                    st.session_state.vectorstore_path = vpath
                    go_to("quiz")

    with tab2:
        ozet = genel_ozet(st.session_state.ogrenci_id)
        if ozet["deneme_sayisi"] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam Deneme", ozet["deneme_sayisi"])
            c2.metric("Ortalama Puan", "%" + str(ozet['ortalama_puan']))
            c3.metric("En Yüksek", "%" + str(ozet['en_yuksek']))
            st.markdown("#### Gelişim Grafiği")
            gelisim = gelisim_takibi(st.session_state.ogrenci_id)
            if gelisim:
                import pandas as pd
                df = pd.DataFrame(gelisim)
                st.line_chart(df.set_index("tarih")["puan"])
            st.markdown("#### Konu Bazlı Başarı")
            analiz = konu_bazli_analiz(st.session_state.ogrenci_id)
            if analiz:
                for konu, veri in analiz.items():
                    st.markdown("**" + konu + "** - %" + str(veri['basari_orani']) + " (" + str(veri['dogru']) + "/" + str(veri['toplam']) + ")")
                    st.progress(veri['basari_orani'] / 100)
            st.markdown("#### Kişisel Öneri")
            st.info(kisisel_oneri_uret(st.session_state.ogrenci_id, st.session_state.ogrenci_adi))
        else:
            st.info("Henüz quiz çözmedin.")

    st.markdown("---")
    if st.button("Çıkış Yap"):
        reset_course_state()
        st.session_state.asama = "giris"
        st.session_state.ogrenci_id = None
        st.session_state.ogrenci_adi = ""
        st.rerun()


# ===== QUIZ =====
elif st.session_state.asama == "quiz":

    quiz = st.session_state.quiz
    dy_sorular = st.session_state.dy_sorular

    toplam_cs = len(quiz)
    toplam_dy = len(dy_sorular)
    toplam = toplam_cs + toplam_dy

    st.markdown("### 📖 " + st.session_state.ders_adi)

    # İlerleme çubuğu placeholder — widget'lar render edildikten sonra güncellenir
    progress_placeholder = st.empty()

    render_concept_map(st.session_state.get("concept_map"))
    st.markdown("---")

    sekmeler = []
    if quiz:
        sekmeler.append(f"📝 Çoktan Seçmeli ({toplam_cs})")
    if dy_sorular:
        sekmeler.append(f"✅ Doğru/Yanlış ({toplam_dy})")

    tabs = st.tabs(sekmeler)
    idx = 0

    if quiz:
        with tabs[idx]:
            for i, soru in enumerate(quiz):
                konu_etiketi = f'<div class="soru-konu">🏷 {soru.get("konu", "")}</div>' if soru.get("konu") else ""
                st.markdown(
                    f'<div class="soru-kart">'
                    f'<div class="soru-no">Soru {i + 1} / {toplam_cs}</div>'
                    f'{konu_etiketi}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{soru['soru']}**")
                secenekler = [h + ") " + m for h, m in soru["secenekler"].items()]

                # Önceki cevabı bul
                onceki = st.session_state.cevaplar.get(i)
                onceki_idx = None
                if onceki:
                    for j, s in enumerate(secenekler):
                        if s.startswith(onceki):
                            onceki_idx = j
                            break

                secim = st.radio(
                    f"soru_{i}",
                    secenekler,
                    index=onceki_idx,
                    label_visibility="collapsed",
                    key=f"radio_{i}"
                )
                # None kontrolü — boş string de falsy olduğu için is not None kullan
                if secim is not None:
                    st.session_state.cevaplar[i] = secim[0]

                st.markdown("---")
        idx += 1

    if dy_sorular:
        with tabs[idx]:
            for i, soru in enumerate(dy_sorular):
                st.markdown(
                    f'<div class="soru-kart">'
                    f'<div class="soru-no">İfade {i + 1} / {toplam_dy}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{soru['ifade']}**")

                onceki_dy = st.session_state.dy_cevaplar.get(i)
                onceki_dy_idx = None
                if onceki_dy is True:
                    onceki_dy_idx = 0
                elif onceki_dy is False:
                    onceki_dy_idx = 1

                secim = st.radio(
                    f"dy_{i}",
                    ["Doğru", "Yanlış"],
                    index=onceki_dy_idx,
                    label_visibility="collapsed",
                    key=f"dy_radio_{i}"
                )
                if secim is not None:
                    st.session_state.dy_cevaplar[i] = (secim == "Doğru")

                st.markdown("---")

    # Tüm widget'lar render edildikten SONRA ilerlemeyi hesapla
    cevaplanan_cs, cevaplanan_dy = cevaplanan_sayisi_hesapla(toplam_cs, toplam_dy)
    cevaplanan = cevaplanan_cs + cevaplanan_dy

    if toplam > 0:
        oran = cevaplanan / toplam
        progress_placeholder.progress(oran, text=f"Cevaplanan: {cevaplanan} / {toplam}")

    # Gönder butonu
    c1, c2 = st.columns(2)
    with c1:
        eksik = toplam - cevaplanan
        buton_label = "Quizi Gönder" if eksik == 0 else f"Quizi Gönder ({eksik} soru boş)"
        if st.button(buton_label, use_container_width=True):
            if quiz:
                deneme_kaydet(
                    st.session_state.ogrenci_id,
                    st.session_state.ders_adi,
                    quiz,
                    st.session_state.cevaplar
                )
            go_to("sonuc")
    with c2:
        if st.button("Ana Sayfa", use_container_width=True):
            go_to("ders_sec")


# ===== SONUÇ =====
elif st.session_state.asama == "sonuc":
    quiz = st.session_state.quiz
    dy_sorular = st.session_state.dy_sorular
    cevaplar = st.session_state.cevaplar
    dy_cevaplar = st.session_state.dy_cevaplar

    st.markdown("### 🏁 " + st.session_state.ders_adi + " — Sonuçlar")

    cs_dogru = dy_dogru_sayi = 0
    cs_puan = dy_puan = 0

    if quiz:
        cs_dogru = sum(1 for i, s in enumerate(quiz) if cevaplar.get(i) == s["dogru_cevap"])
        cs_puan = int((cs_dogru / len(quiz)) * 100)
    if dy_sorular:
        dy_dogru_sayi = sum(1 for i, s in enumerate(dy_sorular) if dy_cevaplar.get(i) == s["dogru_mu"])
        dy_puan = int((dy_dogru_sayi / len(dy_sorular)) * 100)

    toplam_soru = len(quiz) + len(dy_sorular)
    toplam_dogru = cs_dogru + dy_dogru_sayi
    genel_puan = int((toplam_dogru / toplam_soru) * 100) if toplam_soru > 0 else 0

    if genel_puan >= 80:
        st.success("🎉 Harika! Konuyu çok iyi kavradın.")
    elif genel_puan >= 60:
        st.warning("👍 İyi gidiyorsun, birkaç konuyu tekrar et.")
    else:
        st.error("📚 Konuyu tekrar gözden geçir.")

    if quiz and dy_sorular:
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Genel Puan", f"%{genel_puan}")
        c2.metric("📝 Çoktan Seçmeli", f"%{cs_puan}  ({cs_dogru}/{len(quiz)})")
        c3.metric("✅ Doğru/Yanlış", f"%{dy_puan}  ({dy_dogru_sayi}/{len(dy_sorular)})")
    elif quiz:
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Puan", f"%{cs_puan}")
        c2.metric("✔ Doğru", f"{cs_dogru}/{len(quiz)}")
        c3.metric("✘ Yanlış", f"{len(quiz) - cs_dogru}/{len(quiz)}")
    elif dy_sorular:
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Puan", f"%{dy_puan}")
        c2.metric("✔ Doğru", f"{dy_dogru_sayi}/{len(dy_sorular)}")
        c3.metric("✘ Yanlış", f"{len(dy_sorular) - dy_dogru_sayi}/{len(dy_sorular)}")

    st.markdown("---")

    # ---- ÇOKTAN SEÇMELİ DETAY ----
    if quiz:
        st.markdown("### 📝 Çoktan Seçmeli — Cevap Analizi")
        try:
            with st.spinner("Cevaplar analiz ediliyor..."):
                vectorstore = load_vectorstore(st.session_state.vectorstore_path)
                analizler = tum_cevaplari_analiz_et(vectorstore, quiz, cevaplar)
            for a in analizler:
                if a["dogru_mu"]:
                    st.markdown(f'<div class="fb-dogru">✔ Soru {a["soru_no"]}: {a["soru"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="fb-yanlis">✘ Soru {a["soru_no"]}: {a["soru"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fb-aciklama">{a["analiz"]}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning("Cevap analizi oluşturulamadı: " + str(e))

    # ---- DOĞRU/YANLIŞ DETAY ----
    if dy_sorular:
        st.markdown("---")
        st.markdown("### ✅ Doğru/Yanlış — Cevap Analizi")
        for i, soru in enumerate(dy_sorular):
            kullanici = dy_cevaplar.get(i)
            dogru_cevap = soru["dogru_mu"]
            dt = "Doğru" if dogru_cevap else "Yanlış"
            if kullanici == dogru_cevap:
                st.markdown(f'<div class="fb-dogru">✔ İfade {i + 1}: {soru["ifade"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="fb-yanlis">✘ İfade {i + 1}: {soru["ifade"]} &nbsp;(Doğru cevap: {dt})</div>', unsafe_allow_html=True)
            if soru.get("aciklama"):
                st.markdown(f'<div class="fb-aciklama">{soru["aciklama"]}</div>', unsafe_allow_html=True)

    # ---- ÖNERİLER ----
    st.markdown("---")
    st.markdown("### 🎯 Sana Özel Öğrenme Önerileri")
    try:
        oneri = ders_onerisi_uret(st.session_state.ogrenci_id)
        st.info(oneri["mesaj"])
        for o in oneri.get("oneriler", []):
            with st.expander("📌 " + o['konu']):
                st.markdown("**Neden önemli:** " + o['onem'])
                st.markdown("**Nasıl çalışmalı:** " + o['calisma_yontemi'])
                st.markdown("**Kaynak:** " + o['kaynak'])
    except Exception as e:
        st.warning("Öneri üretilemedi: " + str(e))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Tekrar Çöz", use_container_width=True):
            st.session_state.cevaplar = {}
            st.session_state.dy_cevaplar = {}
            go_to("quiz")
    with c2:
        if st.button("🏠 Ana Sayfaya Dön", use_container_width=True):
            go_to("ders_sec")