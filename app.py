import streamlit as st
import numpy as np
import json
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NutriLens",
    page_icon="🍱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0a; color: #f0ede6; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 680px; }

.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-logo { font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800; letter-spacing: -2px; color: #f0ede6; line-height: 1; }
.hero-logo span { color: #c8f060; }
.hero-tag { font-size: 0.85rem; font-weight: 300; color: #888; letter-spacing: 3px; text-transform: uppercase; margin-top: 0.5rem; }

.upload-zone { border: 1.5px dashed #333; border-radius: 16px; padding: 2.5rem; text-align: center; margin: 1.5rem 0; background: #111; transition: border-color 0.2s; }
.upload-zone:hover { border-color: #c8f060; }

.result-card { background: #111; border: 1px solid #222; border-radius: 20px; padding: 2rem; margin: 1.5rem 0; }
.food-name { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #f0ede6; margin: 0; }
.confidence-badge { display: inline-block; background: #c8f060; color: #0a0a0a; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; padding: 4px 12px; border-radius: 100px; margin-top: 0.5rem; }

.nutri-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.5rem; }
.nutri-item { background: #1a1a1a; border-radius: 14px; padding: 1.2rem 0.8rem; text-align: center; }
.nutri-value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #c8f060; line-height: 1; }
.nutri-unit { font-size: 0.7rem; color: #666; margin-top: 2px; }
.nutri-label { font-size: 0.75rem; color: #888; margin-top: 0.4rem; font-weight: 500; }

.alt-item { display: flex; justify-content: space-between; align-items: center; padding: 0.8rem 0; border-bottom: 1px solid #1e1e1e; }
.alt-name { font-size: 0.9rem; color: #aaa; }
.alt-conf { font-size: 0.85rem; color: #555; font-family: 'Syne', sans-serif; }

.conf-bar-bg { height: 4px; background: #1e1e1e; border-radius: 2px; margin-top: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; background: #c8f060; border-radius: 2px; }

.diet-card { background: #111; border: 1px solid #222; border-left: 3px solid #c8f060; border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem; font-size: 0.88rem; color: #aaa; line-height: 1.6; }

.disclaimer { text-align: center; font-size: 0.75rem; color: #444; margin-top: 3rem; padding-bottom: 2rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Confidence threshold ──────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60

# ── Load model & assets ───────────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model('best_nutritionist_model.keras')
    return model

@st.cache_data
def load_assets():
    with open('assets.json', 'r') as f:
        return json.load(f)

# ── Nutrition table ───────────────────────────────────────────
NUTRITION_TABLE = {
    'air':         {'kalori': 0,   'protein': 0.0,  'karbohidrat': 0.0,  'lemak': 0.0},
    'kopi':        {'kalori': 2,   'protein': 0.3,  'karbohidrat': 0.0,  'lemak': 0.0},
    'anggur':      {'kalori': 69,  'protein': 0.7,  'karbohidrat': 18.1, 'lemak': 0.2},
    'apel':        {'kalori': 52,  'protein': 0.3,  'karbohidrat': 13.8, 'lemak': 0.2},
    'durian':      {'kalori': 147, 'protein': 1.5,  'karbohidrat': 27.1, 'lemak': 5.3},
    'jeruk':       {'kalori': 47,  'protein': 0.9,  'karbohidrat': 11.8, 'lemak': 0.1},
    'ayam':        {'kalori': 239, 'protein': 27.3, 'karbohidrat': 0.0,  'lemak': 13.6},
    'cumi':        {'kalori': 175, 'protein': 18.0, 'karbohidrat': 7.8,  'lemak': 7.5},
    'ikan':        {'kalori': 196, 'protein': 22.0, 'karbohidrat': 0.0,  'lemak': 11.5},
    'bakso':       {'kalori': 156, 'protein': 9.7,  'karbohidrat': 10.3, 'lemak': 8.2},
    'bakwan':      {'kalori': 218, 'protein': 4.8,  'karbohidrat': 22.5, 'lemak': 12.6},
    'batagor':     {'kalori': 232, 'protein': 8.2,  'karbohidrat': 23.3, 'lemak': 11.9},
    'bubur':       {'kalori': 124, 'protein': 9.1,  'karbohidrat': 14.4, 'lemak': 3.1},
    'cakwe':       {'kalori': 316, 'protein': 7.4,  'karbohidrat': 38.2, 'lemak': 15.3},
    'capcay':      {'kalori': 67,  'protein': 3.8,  'karbohidrat': 7.6,  'lemak': 2.4},
    'fu yung hai': {'kalori': 154, 'protein': 8.2,  'karbohidrat': 9.1,  'lemak': 9.3},
    'gudeg':       {'kalori': 180, 'protein': 5.4,  'karbohidrat': 22.1, 'lemak': 7.9},
    'kacang':      {'kalori': 567, 'protein': 25.8, 'karbohidrat': 16.1, 'lemak': 49.2},
    'kerupuk':     {'kalori': 476, 'protein': 1.0,  'karbohidrat': 71.0, 'lemak': 21.0},
    'burger':      {'kalori': 258, 'protein': 13.2, 'karbohidrat': 24.1, 'lemak': 11.8},
    'crepes':      {'kalori': 193, 'protein': 5.9,  'karbohidrat': 25.2, 'lemak': 7.7},
    'donat':       {'kalori': 350, 'protein': 4.9,  'karbohidrat': 44.6, 'lemak': 17.2},
    'es krim':     {'kalori': 207, 'protein': 3.5,  'karbohidrat': 23.6, 'lemak': 11.0},
    'kebab':       {'kalori': 224, 'protein': 12.8, 'karbohidrat': 22.1, 'lemak': 9.4},
    'kentang':     {'kalori': 312, 'protein': 3.4,  'karbohidrat': 36.5, 'lemak': 17.0},
}

def get_nutrition(food_name):
    key = food_name.lower().strip().replace('_', ' ')
    return NUTRITION_TABLE.get(key, {
        'kalori': 0, 'protein': 0, 'karbohidrat': 0, 'lemak': 0
    })

def get_diet_tip(food_name, kalori):
    tips = {
        'air':         "Air putih tidak mengandung kalori. Konsumsi 8 gelas per hari untuk hidrasi optimal.",
        'kopi':        "Kopi hitam sangat rendah kalori. Hindari tambahan gula atau krimer berlebih.",
        'anggur':      "Anggur kaya antioksidan resveratrol. Porsi ideal 15-20 butir per hari.",
        'apel':        "Apel tinggi serat dan rendah kalori — camilan ideal untuk diet.",
        'durian':      "Durian tinggi kalori dan karbohidrat. Batasi konsumsi jika sedang diet.",
        'jeruk':       "Jeruk kaya vitamin C dan rendah kalori. Baik dikonsumsi pagi hari.",
        'ayam':        "Ayam adalah sumber protein berkualitas tinggi. Pilih bagian dada tanpa kulit untuk kalori lebih rendah.",
        'cumi':        "Cumi kaya protein dan rendah lemak jenuh. Hindari metode penggorengan berlebih.",
        'ikan':        "Ikan adalah sumber omega-3 terbaik. Direkomendasikan 2-3 porsi per minggu.",
        'bakso':       "Bakso mengandung protein cukup baik. Perhatikan kandungan sodium dari kuahnya.",
        'bakwan':      "Bakwan digoreng, cukup tinggi lemak. Batasi frekuensi konsumsi.",
        'batagor':     "Batagor mengandung protein dari ikan. Kurangi bumbu kacang untuk menurunkan kalori.",
        'bubur':       "Bubur mudah dicerna dan relatif rendah kalori. Pilihan baik untuk sarapan.",
        'cakwe':       "Cakwe tinggi karbohidrat dan lemak karena digoreng. Konsumsi secukupnya.",
        'capcay':      "Capcay rendah kalori dan kaya serat. Salah satu pilihan terbaik untuk diet.",
        'fu yung hai': "Fu Yung Hai mengandung protein dari telur. Perhatikan saus yang digunakan.",
        'gudeg':       "Gudeg kaya karbohidrat dari nangka muda. Porsi seimbang dengan protein.",
        'kacang':      "Kacang sangat tinggi kalori tapi kaya nutrisi. Cukup segenggam per hari.",
        'kerupuk':     "Kerupuk sangat tinggi kalori dan rendah nutrisi. Sebaiknya dikurangi.",
        'burger':      "Burger cukup tinggi kalori. Pilih versi tanpa saus berlebih.",
        'crepes':      "Crepes moderat kalori. Pilih topping buah daripada coklat untuk versi lebih sehat.",
        'donat':       "Donat tinggi gula dan lemak. Konsumsi sesekali sebagai treat.",
        'es krim':     "Es krim tinggi gula dan lemak jenuh. Pilih porsi kecil atau versi rendah lemak.",
        'kebab':       "Kebab mengandung protein cukup baik. Kurangi saus mayones untuk kalori lebih rendah.",
        'kentang':     "Kentang goreng tinggi lemak. Pilih kentang rebus atau panggang sebagai alternatif sehat.",
    }
    return tips.get(food_name.lower().strip().replace('_', ' '),
                    f"Perhatikan porsi konsumsi — {food_name} mengandung {kalori} kcal per 100g.")


def preprocess_image(img, target_size=(300, 300)):
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = img.convert('RGB').resize(target_size)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(np.expand_dims(arr, 0))
    return arr


def predict(model, img_array, class_names, top_k=3):
    probs   = model.predict(img_array, verbose=0)[0]
    indices = np.argsort(probs)[::-1][:top_k]
    top_conf = float(probs[indices[0]])
    results = []
    for idx in indices:
        name  = class_names[idx]
        conf  = float(probs[idx])
        nutri = get_nutrition(name)
        results.append({
            'name':       name,
            'label':      name.replace('_', ' ').title(),
            'confidence': conf,
            **nutri
        })
    # Tandai apakah prediksi teratas cukup yakin
    results[0]['is_confident'] = top_conf >= CONFIDENCE_THRESHOLD
    return results


# ── UI ────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-logo">Nutri<span>Lens</span></div>
    <div class="hero-tag">AI · Food Recognition · Indonesia</div>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    "Foto makananmu",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem">📸</div>
        <div style="font-size:0.95rem; color:#666; margin-top:0.5rem">
            Upload foto makanan untuk analisis nutrisi
        </div>
        <div style="font-size:0.75rem; color:#444; margin-top:0.3rem">
            JPG / PNG · Supports 25 jenis makanan Indonesia
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Daftar makanan yang didukung ──
    with st.expander("📋 Lihat 25 makanan yang didukung"):
        supported = [
            "🥤 Air", "☕ Kopi", "🍇 Anggur", "🍎 Apel", "🍈 Durian",
            "🍊 Jeruk", "🍗 Ayam", "🦑 Cumi", "🐟 Ikan", "🍢 Bakso",
            "🥘 Bakwan", "🥟 Batagor", "🍚 Bubur", "🥐 Cakwe", "🥦 Capcay",
            "🍳 Fu Yung Hai", "🫙 Gudeg", "🥜 Kacang", "🍘 Kerupuk", "🍔 Burger",
            "🥞 Crepes", "🍩 Donat", "🍦 Es Krim", "🌯 Kebab", "🍟 Kentang"
        ]
        cols = st.columns(3)
        for i, item in enumerate(supported):
            cols[i % 3].markdown(f"<small>{item}</small>", unsafe_allow_html=True)

else:
    # Tampilkan gambar
    img = Image.open(uploaded)
    st.image(img, use_container_width=True, caption="", output_format="JPEG")

    with st.spinner("Menganalisis makanan..."):
        try:
            model       = load_model()
            assets      = load_assets()
            CLASS_NAMES = assets['class_names']
        except Exception:
            st.error("⚠️ Model tidak ditemukan. Pastikan file `best_nutritionist_model.keras` dan `assets.json` ada di folder yang sama.")
            st.stop()

        img_array = preprocess_image(img)
        results   = predict(model, img_array, CLASS_NAMES)

    top = results[0]
    import streamlit.components.v1 as components

    # ── Confidence check ──
    if not top['is_confident']:
        components.html("""
        <div style="
            background: #1a1000;
            border: 1px solid #f59e0b;
            border-left: 3px solid #f59e0b;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin: 0.5rem 0;
            color: #f59e0b;
            font-size: 0.88rem;
            font-family: 'DM Sans', sans-serif;
        ">
            ⚠️ <strong>Makanan tidak dikenali dengan yakin.</strong><br>
            <span style="color:#888; font-size:0.82rem">
                NutriLens hanya mengenali 25 jenis makanan. Pastikan foto jelas dan makanan
                termasuk dalam daftar yang didukung.
            </span>
        </div>
        """, height=90)
    else:
        # ── Result card ──
        conf_pct = f"{top['confidence']:.0%}"
        tip = get_diet_tip(top['name'], top['kalori'])
        components.html(f"""
        <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            body {{ margin: 0; background: transparent; }}
            .result-card {{
                background: #111;
                border: 1px solid #222;
                border-radius: 20px;
                padding: 1.8rem;
                margin-bottom: 1rem;
                font-family: 'DM Sans', sans-serif;
            }}
            .food-name {{
                font-family: 'Syne', sans-serif;
                font-size: 2rem;
                font-weight: 700;
                color: #f0ede6;
                margin: 0 0 0.4rem 0;
            }}
            .confidence-badge {{
                display: inline-block;
                background: #c8f060;
                color: #0a0a0a;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 1px;
                padding: 4px 12px;
                border-radius: 100px;
            }}
            .nutri-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1rem;
                margin-top: 1.5rem;
            }}
            .nutri-item {{
                background: #1a1a1a;
                border-radius: 14px;
                padding: 1.2rem 0.8rem;
                text-align: center;
            }}
            .nutri-value {{
                font-family: 'Syne', sans-serif;
                font-size: 1.6rem;
                font-weight: 700;
                color: #c8f060;
                line-height: 1;
            }}
            .nutri-unit {{ font-size: 0.7rem; color: #666; margin-top: 2px; }}
            .nutri-label {{ font-size: 0.75rem; color: #888; margin-top: 0.4rem; font-weight: 500; }}
            .diet-card {{
                background: #111;
                border: 1px solid #222;
                border-left: 3px solid #c8f060;
                border-radius: 12px;
                padding: 1.2rem 1.5rem;
                font-size: 0.88rem;
                color: #aaa;
                line-height: 1.6;
                font-family: 'DM Sans', sans-serif;
            }}
        </style>

        <div class="result-card">
            <p class="food-name">{top['label']}</p>
            <span class="confidence-badge">KEYAKINAN {conf_pct}</span>
            <div class="nutri-grid">
                <div class="nutri-item">
                    <div class="nutri-value">{top['kalori']}</div>
                    <div class="nutri-unit">kcal</div>
                    <div class="nutri-label">Kalori</div>
                </div>
                <div class="nutri-item">
                    <div class="nutri-value">{top['protein']}</div>
                    <div class="nutri-unit">gram</div>
                    <div class="nutri-label">Protein</div>
                </div>
                <div class="nutri-item">
                    <div class="nutri-value">{top['karbohidrat']}</div>
                    <div class="nutri-unit">gram</div>
                    <div class="nutri-label">Karbo</div>
                </div>
                <div class="nutri-item">
                    <div class="nutri-value">{top['lemak']}</div>
                    <div class="nutri-unit">gram</div>
                    <div class="nutri-label">Lemak</div>
                </div>
            </div>
        </div>

        <div class="diet-card">
            💡 <strong>Saran:</strong> {tip}
        </div>
        """, height=340)

    # ── Top-3 alternatif (selalu ditampilkan) ──
    st.markdown("**Kemungkinan lain:**")

    alts_html = ""
    for r in results[1:]:
        bar_w = int(r['confidence'] * 100)
        alts_html += f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.8rem 0; border-bottom:1px solid #1e1e1e;">
            <div style="flex:1">
                <div style="font-size:0.9rem; color:#aaa; font-family:'DM Sans',sans-serif">{r['label']}</div>
                <div style="height:4px; background:#1e1e1e; border-radius:2px; margin-top:6px; overflow:hidden;">
                    <div style="height:100%; width:{bar_w}%; background:#c8f060; border-radius:2px;"></div>
                </div>
            </div>
            <div style="font-size:0.85rem; color:#555; font-family:'Syne',sans-serif; margin-left:1rem">
                {r['confidence']:.1%}
            </div>
        </div>
        """

    import streamlit.components.v1 as components
    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Sans:wght@400&display=swap" rel="stylesheet">
    <div style="background:#111; border:1px solid #222; border-radius:20px; padding:1rem 1.5rem;">
        {alts_html}
    </div>
    """, height=160)

# ── Disclaimer ────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    Data nutrisi per 100g · Sumber: TKPI Kemenkes RI & USDA FoodData<br>
    NutriLens bukan alat diagnosis medis.<br>
    Konsultasikan kebutuhan gizi dengan ahli gizi atau dokter.
</div>
""", unsafe_allow_html=True)