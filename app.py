import streamlit as st
import os
import shutil
import subprocess
import tempfile
from modules.transcription import transcribe_audio, clean_transcript, load_transcript, split_transcript
from modules.rag import create_vectorstore
from modules.quiz import generate_quiz
from modules.concept_map import extract_concept_map

st.set_page_config(
    page_title="SmartEdu",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎓 SmartEdu</h1>
    <p>Yapay Zeka Destekli Öğrenci Değerlendirme Sistemi</p>
</div>
""", unsafe_allow_html=True)

# Session state
for key, val in {
    "aşama": "yukle",
    "quiz": [],
    "cevaplar": {},
    "gonderildi": False,
    "ders_adi": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


def download_from_url(url: str, output_dir: str = "data/audio") -> str:
    """
    Herhangi bir platformdan ses indirir.
    YouTube, Google Drive, Zoom Cloud vb. destekler.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "%(title)s.%(ext)s")

    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_path,
        url
    ]

    subprocess.run(command, check=True)

    for file in os.listdir(output_dir):
        if file.endswith(".mp3"):
            return os.path.join(output_dir, file)

    raise FileNotFoundError("Ses dosyası indirilemedi.")


# AŞAMA 1: URL Girişi
if st.session_state.aşama == "yukle":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📹 Ders Kaydı Ekle")

        ders_adi = st.text_input(
            "Ders Adı",
            placeholder="Örn: Java Programlama - Hafta 3"
        )

        platform = st.selectbox(
            "Platform seç",
            ["YouTube", "Google Drive", "Microsoft Teams (Drive üzerinden)", "Zoom Cloud", "Diğer URL"]
        )

        platform_ipuclari = {
            "YouTube": "https://www.youtube.com/watch?v=...",
            "Google Drive": "https://drive.google.com/file/d/.../view",
            "Microsoft Teams (Drive üzerinden)": "https://drive.google.com/...",
            "Zoom Cloud": "https://zoom.us/rec/share/...",
            "Diğer URL": "https://..."
        }

        if platform == "Microsoft Teams (Drive üzerinden)":
            st.info("💡 Teams kayıtlarını önce OneDrive veya Google Drive'a yükleyip linkini buraya yapıştırın.")

        url = st.text_input(
            "Video URL",
            placeholder=platform_ipuclari[platform]
        )

        soru_sayisi = st.slider(
            "Quiz soru sayısı",
            min_value=3,
            max_value=15,
            value=8
        )

        if url and ders_adi:
            if st.button("🚀 Analiz Et ve Quiz Oluştur", use_container_width=True):
                ders_key = ders_adi.replace(" ", "_").lower()

                # Adım 1: İndir
                with st.status("⬇️ Video indiriliyor...", expanded=True) as status:
                    try:
                        # Eski ses dosyalarını temizle
                        if os.path.exists("data/audio"):
                            shutil.rmtree("data/audio")
                        
                        audio_path = download_from_url(url)
                        status.update(label="✅ Video indirildi!", state="complete")
                    except Exception as e:
                        st.error(f"İndirme hatası: {e}")
                        st.stop()

                # Adım 2: Transkript
                with st.status("🎙️ Ses transkripte çevriliyor...", expanded=True) as status:
                    st.write("Bu işlem birkaç dakika sürebilir...")
                    raw_transcript = transcribe_audio(audio_path)
                    clean_text = clean_transcript(raw_transcript)

                    os.makedirs("data/transcripts", exist_ok=True)
                    transcript_path = f"data/transcripts/{ders_key}.txt"
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        f.write(clean_text)

                    status.update(label="✅ Transkript hazır!", state="complete")

                # Adım 3: RAG
                with st.status("🧠 İçerik analiz ediliyor...", expanded=True) as status:
                    documents = load_transcript(transcript_path)
                    chunks = split_transcript(documents)
                    st.write(f"{len(chunks)} parça oluşturuldu.")

                    vectorstore_path = f"data/vectorstore/{ders_key}"
                    if os.path.exists(vectorstore_path):
                        shutil.rmtree(vectorstore_path)
                    vectorstore = create_vectorstore(chunks, vectorstore_path)

                    status.update(label="✅ Analiz tamamlandı!", state="complete")



                # Adım 3.5: Kavram Haritası
                with st.status("🗺️ Kavram haritası oluşturuluyor...", expanded=True) as status:
                    concept_map = extract_concept_map(vectorstore)
                    st.session_state.concept_map = concept_map
                    status.update(label="✅ Kavram haritası hazır!", state="complete")

                # Adım 4: Quiz
                with st.status("📝 Quiz üretiliyor...", expanded=True) as status:
                    quiz = generate_quiz(vectorstore, soru_sayisi=soru_sayisi)
                    status.update(label=f"✅ {len(quiz)} soru üretildi!", state="complete")

                st.session_state.quiz = quiz
                st.session_state.cevaplar = {}
                st.session_state.gonderildi = False
                st.session_state.aşama = "quiz"
                st.session_state.ders_adi = ders_adi
                st.session_state.vectorstore_path = vectorstore_path

                st.rerun()

# AŞAMA 2: Quiz
elif st.session_state.aşama == "quiz":
    st.markdown(f"### 📝 {st.session_state.ders_adi} - Quiz")
    st.markdown(f"Toplam **{len(st.session_state.quiz)}** soru")
    st.markdown("---")

    quiz = st.session_state.quiz

    for i, soru in enumerate(quiz):
        st.markdown(f"**Soru {i+1}:** {soru['soru']}")

        secenekler = [
            f"{harf}) {metin}"
            for harf, metin in soru['secenekler'].items()
        ]

        secim = st.radio(
            label=f"Soru {i+1}",
            options=secenekler,
            index=None,
            label_visibility="collapsed",
            key=f"radio_{i}"
        )

        if secim:
            st.session_state.cevaplar[i] = secim[0]

        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Quizi Gönder", use_container_width=True):
            st.session_state.aşama = "sonuc"
            st.rerun()
    with col2:
        if st.button("🔄 Yeni Ders", use_container_width=True):
            st.session_state.aşama = "yukle"
            st.rerun()

# AŞAMA 3: Sonuçlar
elif st.session_state.aşama == "sonuc":
    st.markdown(f"### 📊 {st.session_state.ders_adi} - Sonuçlar")

    quiz = st.session_state.quiz
    dogru_sayisi = 0
    yanlis_konular = []

    for i, soru in enumerate(quiz):
        kullanici = st.session_state.cevaplar.get(i)
        dogru = soru["dogru_cevap"]

        if kullanici == dogru:
            st.success(f"✅ Soru {i+1}: Doğru! ({dogru})")
            dogru_sayisi += 1
        elif kullanici:
            st.error(f"❌ Soru {i+1}: Yanlış! Senin cevabın: **{kullanici}** | Doğru: **{dogru}**")
            st.info(f"💡 {soru['aciklama']}")
            yanlis_konular.append(soru['soru'])
        else:
            st.warning(f"⚠️ Soru {i+1}: Cevaplanmadı. Doğru: **{dogru}**")
            yanlis_konular.append(soru['soru'])

    if "concept_map" in st.session_state and st.session_state.concept_map:
        st.markdown("---")
        st.markdown("### 🗺️ Ders Kavram Haritası")

        cm = st.session_state.concept_map
        st.markdown(f"**📚 {cm.get('ders_konusu', '')}**")

        for idx, ana in enumerate(cm.get("ana_konular", []), 1):
            with st.expander(f"📌 {idx}. {ana['baslik']}", expanded=True):
                st.markdown(f"_{ana['aciklama']}_")
                for alt in ana.get("alt_konular", []):
                    st.markdown(f"- **{alt['baslik']}**: {alt['aciklama']}")

    st.markdown("---")

    puan = (dogru_sayisi / len(quiz)) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Puanın", f"{puan:.0f}%")
    with col2:
        st.metric("Doğru", f"{dogru_sayisi}/{len(quiz)}")
    with col3:
        st.metric("Yanlış", f"{len(quiz)-dogru_sayisi}/{len(quiz)}")

    if puan >= 80:
        st.balloons()
        st.success("🎉 Harika! Konuyu çok iyi kavradın!")
    elif puan >= 60:
        st.warning("👍 İyi gidiyorsun, biraz daha çalışabilirsin!")
    else:
        st.error("📚 Konuyu tekrar gözden geçirmeni öneririm!")

    if yanlis_konular:
        st.markdown("### 📌 Tekrar Çalışman Gereken Konular")
        for konu in yanlis_konular:
            st.markdown(f"- {konu}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Tekrar Dene", use_container_width=True):
            st.session_state.cevaplar = {}
            st.session_state.aşama = "quiz"
            st.rerun()
    with col2:
        if st.button("📹 Yeni Ders Yükle", use_container_width=True):
            st.session_state.aşama = "yukle"
            st.rerun()