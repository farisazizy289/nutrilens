# 🍱 NutriLens

> AI-powered food recognition & nutrition analyzer for Indonesian cuisine

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

NutriLens adalah aplikasi web berbasis AI yang dapat mengenali 25 jenis makanan Indonesia dari foto dan langsung menampilkan informasi nutrisinya — kalori, protein, karbohidrat, dan lemak per 100g.

---

## ✨ Fitur

- 📸 **Upload foto** makanan (JPG/PNG)
- 🤖 **Klasifikasi otomatis** menggunakan model EfficientNetB3 fine-tuned
- 📊 **Informasi nutrisi** lengkap per 100g (kalori, protein, karbo, lemak)
- 💡 **Saran diet** personal per jenis makanan
- ⚠️ **Confidence threshold** — peringatan jika makanan tidak dikenali dengan yakin (< 60%)
- 🔁 **Top-3 prediksi** alternatif dengan confidence bar
- 📋 **Daftar 25 makanan** yang didukung selalu terlihat

---

## 🍽️ Makanan yang Didukung

| | | |
|---|---|---|
| 🥤 Air | ☕ Kopi | 🍇 Anggur |
| 🍎 Apel | 🍈 Durian | 🍊 Jeruk |
| 🍗 Ayam | 🦑 Cumi | 🐟 Ikan |
| 🍢 Bakso | 🥘 Bakwan | 🥟 Batagor |
| 🍚 Bubur | 🥐 Cakwe | 🥦 Capcay |
| 🍳 Fu Yung Hai | 🫙 Gudeg | 🥜 Kacang |
| 🍘 Kerupuk | 🍔 Burger | 🥞 Crepes |
| 🍩 Donat | 🍦 Es Krim | 🌯 Kebab |
| 🍟 Kentang | | |

---

## 🧠 Model

- **Arsitektur:** EfficientNetB3 (fine-tuned)
- **Input size:** 300 × 300 px
- **Preprocessing:** `EfficientNet preprocess_input` dari Keras
- **Output:** Softmax 25 kelas
- **File model:** `best_nutritionist_model.keras`

---

## 🗂️ Struktur Proyek

```
nutrilens/
├── app.py                        # Aplikasi Streamlit utama
├── best_nutritionist_model.keras # Model EfficientNetB3 (tidak di-push ke repo)
├── assets.json                   # Daftar class names
├── requirements.txt              # Dependensi Python
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone repo

```bash
git clone https://github.com/username/nutrilens.git
cd nutrilens
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Letakkan file model

Pastikan file berikut ada di root folder:
- `best_nutritionist_model.keras`
- `assets.json` (berisi `{"class_names": [...]}`)

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
tensorflow
numpy
Pillow
```

Atau install langsung:

```bash
pip install streamlit tensorflow numpy Pillow
```

---

## 📊 Sumber Data Nutrisi

Data nutrisi per 100g bersumber dari:
- **TKPI** — Tabel Komposisi Pangan Indonesia, Kemenkes RI
- **USDA FoodData Central**

---

## ⚠️ Disclaimer

NutriLens bukan alat diagnosis medis. Data nutrisi bersifat estimasi per 100g bahan makanan. Konsultasikan kebutuhan gizi Anda dengan ahli gizi atau dokter.

---

## 📄 Lisensi

[MIT License](LICENSE)