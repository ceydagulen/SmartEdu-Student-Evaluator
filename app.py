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
    """Secilen tipe gore quiz uretir."""
    quiz, dy_sorular = [], []
    if tip in ("Coktan Secmeli", "Ikisi de"):
        quiz = generate_quiz(vectorstore, soru_sayisi=cs_sayi)
    if tip in ("Dogru/Yanlis", "Ikisi de"):
        try:
            dy_sorular = generate_true_false(vectorstore, soru_sayisi=dy_sayi)
        except Exception:
            dy_sorular = []
    return quiz, dy_sorular


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
    .soru-no { font-size: 15px; font-weight: 800; color: #764ba2; margin-bottom: 8px; }
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
    <p>Ders videolarini analiz ederek otomatik quiz olusturan, ogrenci cevaplarini degerlendiren ve kisisellestirilmis ogrenme onerileri sunan yapay zeka destekli egitim asistani.</p>
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
    st.markdown("## Sistem Akisi")
    st.markdown("1. Ogrenci girisi\n2. Ders yukleme / secme\n3. Transkript ve analiz\n4. Quiz uretimi\n5. Cevap analizi\n6. Kisisel oneriler")
    st.divider()
    if st.session_state.ogrenci_adi:
        st.success("Ogrenci: " + st.session_state.ogrenci_adi)
    else:
        st.info("Henuz giris yapilmadi.")
    st.caption("AI / NLP / RAG / ChromaDB / SQLite")


# ===== GIRIS =====
if st.session_state.asama == "giris":
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="section-title">Ogrenci Girisi</div>', unsafe_allow_html=True)
        ad = st.text_input("Ad Soyad", placeholder="Orn: Ceyda Gulen")
        ogrenci_no = st.text_input("Ogrenci No", placeholder="Orn: 21360859042")
        if st.button("Giris Yap", use_container_width=True):
            if ad.strip() and ogrenci_no.strip():
                st.session_state.ogrenci_id = ogrenci_ekle_veya_getir(ad.strip(), ogrenci_no.strip())
                st.session_state.ogrenci_adi = ad.strip()
                reset_course_state()
                go_to("ders_sec")
            else:
                st.error("Lutfen ad soyad ve ogrenci numarasini girin.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="info-card"><h4>Ders Analizi</h4><p>Ders videosundan transkript cikarilir ve icerik NLP ile analiz edilir.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="info-card"><h4>Otomatik Quiz</h4><p>Coktan secmeli ve dogru/yanlis sorulari otomatik olusturulur.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="info-card"><h4>Kisisel Oneri</h4><p>Cevaplar analiz edilir ve eksik konulara gore oneriler sunulur.</p></div>', unsafe_allow_html=True)


# ===== DERS SEC =====
elif st.session_state.asama == "ders_sec":
    st.markdown("### Hos geldin, " + st.session_state.ogrenci_adi + "!")
    tab1, tab2 = st.tabs(["Quiz Olustur", "Performansim"])

    with tab1:
        kaynak = st.radio("Ders kaynagi sec:", ["Yeni video yukle", "Kayitli dersten sec"], horizontal=True)

        # --- Quiz tipi ve sayi (ortak) ---
        st.markdown("**Quiz Ayarlari**")
        quiz_tip = st.radio("Quiz tipi:", ["Coktan Secmeli", "Dogru/Yanlis", "Ikisi de"], horizontal=True)
        cc1, cc2 = st.columns(2)
        cs_sayi = dy_sayi = 0
        if quiz_tip in ("Coktan Secmeli", "Ikisi de"):
            cs_sayi = cc1.slider("Coktan secmeli soru sayisi", 5, 30, 10)
        if quiz_tip in ("Dogru/Yanlis", "Ikisi de"):
            dy_sayi = cc2.slider("Dogru/Yanlis soru sayisi", 5, 30, 10)
        st.markdown("---")

        # --- YENI VIDEO ---
        if kaynak == "Yeni video yukle":
            ders_adi = st.text_input("Ders Adi", placeholder="Orn: NumPy Temelleri")
            url = st.text_input("Ders Video URL", placeholder="https://www.youtube.com/watch?v=...")
            st.caption("YouTube veya erisilebilir cevrim ici ders video baglantisi giriniz.")

            if st.button("Analiz Et ve Quiz Olustur", use_container_width=True):
                if not ders_adi.strip() or not url.strip():
                    st.error("Lutfen ders adini ve video baglantisini girin.")
                    st.stop()
                ders_key = slugify(ders_adi)
                st.session_state.ders_adi = ders_adi.strip()
                progress = st.progress(0, text="Islem baslatiliyor...")
                try:
                    with st.status("Video indiriliyor...", expanded=True) as status:
                        if os.path.exists("data/audio"):
                            shutil.rmtree("data/audio")
                        audio_path = download_from_url(url.strip())
                        status.update(label="Video indirildi.", state="complete")
                    progress.progress(20, text="Video indirildi.")
                except Exception as e:
                    st.error("Indirme hatasi: " + str(e)); st.stop()
                try:
                    with st.status("Transkript olusturuluyor...", expanded=True) as status:
                        st.write("Bu islem birkac dakika surebilir.")
                        raw = transcribe_audio(audio_path)
                        clean = clean_transcript(raw)
                        os.makedirs("data/transcripts", exist_ok=True)
                        tpath = "data/transcripts/" + ders_key + ".txt"
                        with open(tpath, "w", encoding="utf-8") as f:
                            f.write(clean)
                        status.update(label="Transkript hazir.", state="complete")
                    progress.progress(45, text="Transkript hazir.")
                except Exception as e:
                    st.error("Transkript hatasi: " + str(e)); st.stop()
                try:
                    with st.status("Icerik analiz ediliyor...", expanded=True) as status:
                        docs = load_transcript(tpath)
                        chunks = split_transcript(docs)
                        vpath = "data/vectorstore/" + ders_key
                        if os.path.exists(vpath):
                            shutil.rmtree(vpath)
                        vectorstore = create_vectorstore(chunks, vpath)
                        status.update(label="Analiz tamamlandi.", state="complete")
                    progress.progress(65, text="Analiz tamamlandi.")
                except Exception as e:
                    st.error("Analiz hatasi: " + str(e)); st.stop()
                try:
                    with st.status("Kavramlar cikariliyor...", expanded=True) as status:
                        concept_map = extract_concept_map(vectorstore)
                        status.update(label="Kavramlar hazir.", state="complete")
                    progress.progress(80, text="Kavramlar cikarildi.")
                except Exception:
                    concept_map = None
                with st.status("Quiz uretiliyor...", expanded=True) as status:
                    quiz, dy_sorular = quiz_uret(vectorstore, quiz_tip, cs_sayi, dy_sayi)
                    status.update(label="Quiz hazir.", state="complete")
                progress.progress(100, text="Tamamlandi.")
                st.session_state.quiz = quiz
                st.session_state.dy_sorular = dy_sorular
                st.session_state.cevaplar = {}
                st.session_state.dy_cevaplar = {}
                st.session_state.concept_map = concept_map
                st.session_state.vectorstore_path = vpath
                go_to("quiz")

        # --- KAYITLI DERS ---
        else:
            kayitli = []
            if os.path.exists("data/vectorstore"):
                kayitli = [d for d in os.listdir("data/vectorstore") if os.path.isdir("data/vectorstore/" + d)]
            if not kayitli:
                st.info("Henuz kayitli ders yok. Once yeni bir video yukleyin.")
            else:
                secili = st.selectbox("Kayitli Ders Sec", kayitli)
                if st.button("Bu Dersten Quiz Olustur", use_container_width=True):
                    vpath = "data/vectorstore/" + secili
                    st.session_state.ders_adi = secili
                    with st.spinner("Quiz olusturuluyor..."):
                        vectorstore = load_vectorstore(vpath)
                        quiz, dy_sorular = quiz_uret(vectorstore, quiz_tip, cs_sayi, dy_sayi)
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
            c3.metric("En Yuksek", "%" + str(ozet['en_yuksek']))
            st.markdown("#### Gelisim Grafigi")
            gelisim = gelisim_takibi(st.session_state.ogrenci_id)
            if gelisim:
                import pandas as pd
                df = pd.DataFrame(gelisim)
                st.line_chart(df.set_index("tarih")["puan"])
            st.markdown("#### Konu Bazli Basari")
            analiz = konu_bazli_analiz(st.session_state.ogrenci_id)
            if analiz:
                for konu, veri in analiz.items():
                    st.markdown("**" + konu + "** - %" + str(veri['basari_orani']) + " (" + str(veri['dogru']) + "/" + str(veri['toplam']) + ")")
                    st.progress(veri['basari_orani'] / 100)
            st.markdown("#### Kisisel Oneri")
            st.info(kisisel_oneri_uret(st.session_state.ogrenci_id, st.session_state.ogrenci_adi))
        else:
            st.info("Henuz quiz cozmedin.")

    st.markdown("---")
    if st.button("Cikis Yap"):
        reset_course_state()
        st.session_state.asama = "giris"
        st.session_state.ogrenci_id = None
        st.session_state.ogrenci_adi = ""
        st.rerun()


# ===== QUIZ =====
elif st.session_state.asama == "quiz":
    st.markdown("### " + st.session_state.ders_adi)
    if st.session_state.get("concept_map"):
        with st.expander("Bu derste one cikan konular"):
            cm = st.session_state.concept_map
            for ana in (cm.get("ana_konular", []) if isinstance(cm, dict) else []):
                st.markdown("**" + ana.get('baslik', '') + "**: " + ana.get('aciklama', ''))

    quiz = st.session_state.quiz
    dy_sorular = st.session_state.dy_sorular

    sekmeler = []
    if quiz:
        sekmeler.append("Coktan Secmeli (" + str(len(quiz)) + ")")
    if dy_sorular:
        sekmeler.append("Dogru/Yanlis (" + str(len(dy_sorular)) + ")")

    tabs = st.tabs(sekmeler)
    idx = 0

    if quiz:
        with tabs[idx]:
            st.markdown("---")
            for i, soru in enumerate(quiz):
                st.markdown('<div class="soru-kart">', unsafe_allow_html=True)
                st.markdown('<div class="soru-no">Soru ' + str(i + 1) + '</div>', unsafe_allow_html=True)
                st.markdown("**" + soru['soru'] + "**")
                secenekler = [h + ") " + m for h, m in soru["secenekler"].items()]
                secim = st.radio("Soru " + str(i + 1), secenekler, index=None, label_visibility="collapsed", key="radio_" + str(i))
                if secim:
                    st.session_state.cevaplar[i] = secim[0]
                st.markdown("</div>", unsafe_allow_html=True)
        idx += 1

    if dy_sorular:
        with tabs[idx]:
            st.markdown("---")
            for i, soru in enumerate(dy_sorular):
                st.markdown('<div class="soru-kart">', unsafe_allow_html=True)
                st.markdown('<div class="soru-no">Ifade ' + str(i + 1) + '</div>', unsafe_allow_html=True)
                st.markdown("**" + soru['ifade'] + "**")
                secim = st.radio("Ifade " + str(i + 1), ["Dogru", "Yanlis"], index=None, label_visibility="collapsed", key="dy_radio_" + str(i))
                if secim:
                    st.session_state.dy_cevaplar[i] = (secim == "Dogru")
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Quizi Gonder", use_container_width=True):
            if quiz:
                deneme_kaydet(st.session_state.ogrenci_id, st.session_state.ders_adi, quiz, st.session_state.cevaplar)
            go_to("sonuc")
    with c2:
        if st.button("Ana Sayfa", use_container_width=True):
            go_to("ders_sec")


# ===== SONUC =====
elif st.session_state.asama == "sonuc":
    st.markdown("### " + st.session_state.ders_adi + " - Sonuclar")
    quiz = st.session_state.quiz
    dy_sorular = st.session_state.dy_sorular
    cevaplar = st.session_state.cevaplar
    dy_cevaplar = st.session_state.dy_cevaplar

    # ---- COKTAN SECMELI SONUC ----
    if quiz:
        st.markdown("## Coktan Secmeli Sonuclari")
        dogru = sum(1 for i, s in enumerate(quiz) if cevaplar.get(i) == s["dogru_cevap"])
        puan = (dogru / len(quiz)) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Puan", "%" + str(int(puan)))
        c2.metric("Dogru", str(dogru) + "/" + str(len(quiz)))
        c3.metric("Yanlis", str(len(quiz) - dogru) + "/" + str(len(quiz)))
        if puan >= 80:
            st.success("Harika! Konuyu cok iyi kavradin.")
        elif puan >= 60:
            st.warning("Iyi gidiyorsun.")
        else:
            st.error("Konuyu tekrar gozden gecir.")

        st.markdown("#### Cevap Analizi")
        try:
            with st.spinner("Cevaplar analiz ediliyor..."):
                vectorstore = load_vectorstore(st.session_state.vectorstore_path)
                analizler = tum_cevaplari_analiz_et(vectorstore, quiz, cevaplar)
            for a in analizler:
                if a["dogru_mu"]:
                    st.markdown('<div class="fb-dogru">Dogru - Soru ' + str(a["soru_no"]) + ': ' + a["soru"] + '</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fb-yanlis">Yanlis - Soru ' + str(a["soru_no"]) + ': ' + a["soru"] + '</div>', unsafe_allow_html=True)
                st.markdown('<div class="fb-aciklama">' + a["analiz"] + '</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning("Cevap analizi olusturulamadi: " + str(e))

    # ---- DOGRU/YANLIS SONUC ----
    if dy_sorular:
        st.markdown("---")
        st.markdown("## Dogru/Yanlis Sonuclari")
        dy_dogru = sum(1 for i, s in enumerate(dy_sorular) if dy_cevaplar.get(i) == s["dogru_mu"])
        dy_puan = (dy_dogru / len(dy_sorular)) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Puan", "%" + str(int(dy_puan)))
        c2.metric("Dogru", str(dy_dogru) + "/" + str(len(dy_sorular)))
        c3.metric("Yanlis", str(len(dy_sorular) - dy_dogru) + "/" + str(len(dy_sorular)))
        for i, soru in enumerate(dy_sorular):
            kullanici = dy_cevaplar.get(i)
            dogru_cevap = soru["dogru_mu"]
            dt = "Dogru" if dogru_cevap else "Yanlis"
            if kullanici == dogru_cevap:
                st.markdown('<div class="fb-dogru">Dogru - Ifade ' + str(i + 1) + ': ' + soru["ifade"] + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fb-yanlis">Yanlis - Ifade ' + str(i + 1) + ': ' + soru["ifade"] + ' (Dogru cevap: ' + dt + ')</div>', unsafe_allow_html=True)
            st.markdown('<div class="fb-aciklama">' + soru.get("aciklama", "") + '</div>', unsafe_allow_html=True)

    # ---- ONERILER ----
    st.markdown("---")
    st.markdown("## Sana Ozel Ogrenme Onerileri")
    try:
        oneri = ders_onerisi_uret(st.session_state.ogrenci_id)
        st.info(oneri["mesaj"])
        for o in oneri.get("oneriler", []):
            with st.expander(o['konu']):
                st.markdown("**Neden onemli:** " + o['onem'])
                st.markdown("**Nasil calismali:** " + o['calisma_yontemi'])
                st.markdown("**Kaynak:** " + o['kaynak'])
    except Exception as e:
        st.warning("Oneri uretilemedi: " + str(e))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Tekrar Coz", use_container_width=True):
            st.session_state.cevaplar = {}
            st.session_state.dy_cevaplar = {}
            go_to("quiz")
    with c2:
        if st.button("Ana Sayfaya Don", use_container_width=True):
            go_to("ders_sec")